def lcm_loss(y_pred, label_sim_dist, y_true, num_classes=4, alpha=4):
    pred_probs = F.softmax(y_pred, dim=-1)
    label_sim_dist = F.softmax(label_sim_dist, dim=-1)
    simulated_y_true = F.log_softmax(label_sim_dist + alpha*F.one_hot(y_true.long(), num_classes=num_classes), dim=1)
    loss = F.kl_div(simulated_y_true, pred_probs, reduction='sum')
    return loss

class LCM_MultiClass(nn.Module):
    def __init__(self, lable_size, emb_dim, output_dim, padding_idx=None):
        super().__init__()
        self.label_emb = nn.Embedding(lable_size, emb_dim, padding_idx=padding_idx)
        self.label_activation = nn.GELU()
        self.label_dnn = nn.Linear(emb_dim, output_dim)
        self.label_fc = nn.Linear(output_dim, lable_size)
    def forward(self, feature, label):
        x = self.label_emb(label)
        x = self.label_dnn(x)
        x = self.label_activation(x)
        x = (x @ feature.unsqueeze(-1)).squeeze(-1)
        x = self.label_fc(x)
        return x
      
 def simcse_loss(y_pred):
    # y_pred.repeat_interleave(2, dim=0)
    idxs = torch.arange(0, y_pred.size(0))  
    idxs_1 = idxs[None, :]  
    idxs_2 = (idxs + 1 - idxs % 2 * 2)[:, None]  
    y_true = idxs_1 == idxs_2
    y_true = y_true.to(torch.float).to(y_pred)

    y_pred = F.normalize(y_pred, dim=1, p=2)
    similarities = torch.matmul(y_pred, y_pred.transpose(0,1))  
    similarities = similarities - torch.eye(y_pred.shape[0]).to(y_pred) * 1e12
    similarities = similarities * 20
    loss = F.binary_cross_entropy_with_logits(similarities, y_true)
    return loss
