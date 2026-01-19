# multitask_legal_model.py - Core Multi-Task Model for Legal Document Analysis

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, List, Optional, Tuple
import json

class MultiTaskLegalModel(nn.Module):
    """Multi-task model for legal document analysis"""
    
    def __init__(self, 
                 model_name: str = "nlpaueb/legal-bert-base-uncased",
                 num_judgment_classes: int = 3,  # Accept, Dismiss, Remand
                 num_section_classes: int = 13,
                 num_argument_classes: int = 5,
                 num_bias_classes: int = 4):
        super().__init__()
        
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Base transformer model
        self.base_model = AutoModel.from_pretrained(model_name)
        
        # Hierarchical document representation
        self.sentence_encoder = nn.LSTM(
            input_size=self.config.hidden_size,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        self.document_encoder = nn.LSTM(
            input_size=1024,  # bidirectional
            hidden_size=512,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Cross-section attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=16,
            dropout=0.2,
            batch_first=True
        )
        
        # Task-specific heads
        
        # 1. Judgment Prediction Head
        self.judgment_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_judgment_classes)
        )
        
        # 2. Section Classification Head
        self.section_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_section_classes)
        )
        
        # 3. Argument Mining Head
        self.argument_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_argument_classes)
        )
        
        # 4. Bias Detection Head
        self.bias_head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_bias_classes)
        )
        
        # 5. Rationale Generation Head (for attention weights)
        self.rationale_attention = nn.Linear(1024, 1)
        
        # Task weight parameters (learnable)
        self.task_weights = nn.Parameter(torch.ones(5))
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                judgment_labels: Optional[torch.Tensor] = None,
                section_labels: Optional[torch.Tensor] = None,
                argument_labels: Optional[torch.Tensor] = None,
                bias_labels: Optional[torch.Tensor] = None,
                return_rationale: bool = True):
        
        batch_size, max_sentences, seq_len = input_ids.shape
        
        # Reshape for transformer processing
        input_ids_flat = input_ids.view(-1, seq_len)
        attention_mask_flat = attention_mask.view(-1, seq_len)
        
        # Get sentence representations from base model
        base_outputs = self.base_model(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat
        )
        
        # Sentence embeddings (mean pooling)
        sentence_embeddings = base_outputs.last_hidden_state.mean(dim=1)  # [batch*sentences, hidden]
        sentence_embeddings = sentence_embeddings.view(batch_size, max_sentences, -1)
        
        # Hierarchical encoding
        sentence_lstm_out, _ = self.sentence_encoder(sentence_embeddings)
        doc_lstm_out, _ = self.document_encoder(sentence_lstm_out)
        
        # Cross-section attention
        attended_output, attention_weights = self.cross_attention(
            doc_lstm_out, doc_lstm_out, doc_lstm_out
        )
        
        # Document representation (mean pooling over sentences)
        doc_representation = attended_output.mean(dim=1)  # [batch, hidden]
        
        # Task predictions
        judgment_logits = self.judgment_head(doc_representation)
        
        # Sentence-level predictions
        section_logits = self.section_head(attended_output)  # [batch, sentences, classes]
        argument_logits = self.argument_head(attended_output)
        
        # Bias detection (document-level)
        bias_logits = self.bias_head(doc_representation)
        
        # Rationale scores (attention weights for important sentences)
        rationale_scores = self.rationale_attention(attended_output).squeeze(-1)  # [batch, sentences]
        
        outputs = {
            'judgment_logits': judgment_logits,
            'section_logits': section_logits,
            'argument_logits': argument_logits,
            'bias_logits': bias_logits,
            'rationale_scores': rationale_scores,
            'attention_weights': attention_weights
        }
        
        # Calculate losses if labels are provided
        if any(labels is not None for labels in [judgment_labels, section_labels, argument_labels, bias_labels]):
            losses = {}
            total_loss = 0
            
            # Judgment prediction loss
            if judgment_labels is not None:
                judgment_loss = F.cross_entropy(judgment_logits, judgment_labels)
                losses['judgment_loss'] = judgment_loss
                total_loss += self.task_weights[0] * judgment_loss
            
            # Section classification loss (sequence labeling)
            if section_labels is not None:
                section_loss = F.cross_entropy(
                    section_logits.view(-1, section_logits.size(-1)),
                    section_labels.view(-1),
                    ignore_index=-100
                )
                losses['section_loss'] = section_loss
                total_loss += self.task_weights[1] * section_loss
            
            # Argument mining loss
            if argument_labels is not None:
                argument_loss = F.cross_entropy(
                    argument_logits.view(-1, argument_logits.size(-1)),
                    argument_labels.view(-1),
                    ignore_index=-100
                )
                losses['argument_loss'] = argument_loss
                total_loss += self.task_weights[2] * argument_loss
            
            # Bias detection loss
            if bias_labels is not None:
                bias_loss = F.cross_entropy(bias_logits, bias_labels)
                losses['bias_loss'] = bias_loss
                total_loss += self.task_weights[3] * bias_loss
            
            # Rationale regularization (encourage sparse attention)
            if return_rationale:
                rationale_reg = torch.norm(rationale_scores, p=1, dim=1).mean()
                losses['rationale_reg'] = rationale_reg
                total_loss += self.task_weights[4] * rationale_reg * 0.01
            
            losses['total_loss'] = total_loss
            outputs['losses'] = losses
        
        return outputs


class MultiTaskTrainer:
    """Trainer for multi-task legal model"""
    
    def __init__(self, model: MultiTaskLegalModel, tokenizer, device):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # Label mappings
        self.judgment_labels = ['ACCEPT', 'DISMISS', 'REMAND']
        self.section_labels = [
            'FACTS', 'ISSUE', 'STATUTES', 'PRECEDENTS', 'ARGUMENTS_PET',
            'ARGUMENTS_RES', 'REASONING', 'FINDINGS', 'ORDERS', 
            'CONCLUSION', 'DISSENT', 'PROCEDURAL', 'OTHER'
        ]
        self.argument_labels = ['CLAIM', 'PREMISE', 'EVIDENCE', 'REBUTTAL', 'OTHER']
        self.bias_labels = ['NO_BIAS', 'GENDER_BIAS', 'ECONOMIC_BIAS', 'SOCIAL_BIAS']
        
    def preprocess_document(self, text: str, max_sentences: int = 128) -> Dict:
        """Preprocess document for multi-task model"""
        import nltk
        from nltk.tokenize import sent_tokenize
        
        # Split into sentences
        sentences = sent_tokenize(text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # Limit sentences
        if len(sentences) > max_sentences:
            sentences = sentences[:max_sentences]
        
        # Tokenize sentences
        tokenized_sentences = []
        for sentence in sentences:
            tokens = self.tokenizer(
                sentence,
                max_length=64,  # Shorter for memory efficiency
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized_sentences.append(tokens)
        
        # Pad to max_sentences
        while len(tokenized_sentences) < max_sentences:
            empty_tokens = self.tokenizer(
                "",
                max_length=64,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            tokenized_sentences.append(empty_tokens)
        
        # Stack tensors
        input_ids = torch.stack([t['input_ids'].squeeze() for t in tokenized_sentences])
        attention_mask = torch.stack([t['attention_mask'].squeeze() for t in tokenized_sentences])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'sentences': sentences,
            'num_sentences': len(sentences)
        }
    
    def predict(self, text: str) -> Dict:
        """Make predictions on new text"""
        self.model.eval()
        
        # Preprocess
        processed = self.preprocess_document(text)
        
        # Prepare input
        input_ids = processed['input_ids'].unsqueeze(0).to(self.device)
        attention_mask = processed['attention_mask'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_rationale=True
            )
        
        # Process predictions
        judgment_pred = torch.argmax(outputs['judgment_logits'], dim=1).cpu().numpy()[0]
        section_preds = torch.argmax(outputs['section_logits'], dim=2).cpu().numpy()[0]
        argument_preds = torch.argmax(outputs['argument_logits'], dim=2).cpu().numpy()[0]
        bias_pred = torch.argmax(outputs['bias_logits'], dim=1).cpu().numpy()[0]
        rationale_scores = torch.softmax(outputs['rationale_scores'], dim=1).cpu().numpy()[0]
        
        # Get top rationale sentences
        import numpy as np
        top_rationale_idx = np.argsort(rationale_scores)[-5:][::-1]
        
        return {
            'judgment_prediction': self.judgment_labels[judgment_pred],
            'section_predictions': [self.section_labels[pred] for pred in section_preds[:len(processed['sentences'])]],
            'argument_predictions': [self.argument_labels[pred] for pred in argument_preds[:len(processed['sentences'])]],
            'bias_prediction': self.bias_labels[bias_pred],
            'rationale_sentences': [processed['sentences'][i] for i in top_rationale_idx if i < len(processed['sentences'])],
            'rationale_scores': rationale_scores[:len(processed['sentences'])].tolist()
        }


# Usage example:
if __name__ == "__main__":
    # Initialize model
    model = MultiTaskLegalModel()
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize trainer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = MultiTaskTrainer(model, tokenizer, device)
    
    print(f"Multi-task model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")