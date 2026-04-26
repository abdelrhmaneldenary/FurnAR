class MyMsa(nn.Module):
    def __init__(self,d,n_heads=2):
        super(MyMsa,self).__init__()
        self.d=d
        self.n_heads=n_heads

        assert d%n_heads==0

        d_head=int(d/n_heads)
        self.q_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.k_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.v_mappings=nn.ModuleList([nn.Linear(d_head,d_head) for _ in range(self.n_heads)])
        self.d_head=d_head
        self.softmax=nn.Softmax(dim=-1)
        self.out_proj=nn.Linear(d,d)

    def forward(self,sequences):
        result=[]
        for sequence in sequences:
            seq_result=[]
            for head in range(self.n_heads):
                q_mapping=self.q_mappings[head]
                k_mapping=self.k_mappings[head]
                v_mapping=self.v_mappings[head]

                seq=sequence[:,head*self.d_head:(head+1)*self.d_head]
                q,k,v=q_mapping(seq),k_mapping(seq),v_mapping(seq)

                attention=self.softmax(q@k.T/(self.d_head**2))
                seq_result.append(attention@v)

            result.append(torch.hstack(seq_result))

        return self.out_proj(torch.cat([torch.unsqueeze(r,dim=0) for r in result]))
                
        