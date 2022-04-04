    def saliency(self, args:namedtuple, N:int=50, conv_num:int=0, 
                       utt_num:int=0, quiet=False):
        """ generate saliency maps for parallel model """

        #method only works for deparallelised system
        #self.deparallelise()
        
        #prepare conversation in interest
        eval_data = self.C.prepare_data(path=args.eval_path, 
                                        lim=args.lim, 
                                        quiet=True)
        conv = eval_data[conv_num]
        convs = self.batcher(data=[conv], bsz=1, shuffle=False)
        conv_b = next(itertools.islice(convs, utt_num, None)) #sellect specific utt

        #Get details of the max prob prediction
        y = self.model_output(conv_b).logits[0]
        pred_idx = torch.argmax(y).item()      
        prob = F.softmax(y, dim=-1)[pred_idx].item()
        
        pred_class = self.C.label_dict[pred_idx]
        true_class = self.C.label_dict[conv_b.labels.item()]
        if not quiet: print(f'pred: {pred_class} ({round(prob, 3)})    ',
                            f'true: {true_class}')
        
        #get InteGrad batches (refer to paper for details)
        tok_attr = self._line_integral(conv.ids, conv_b.utt_pos, quiet)
        
        #get attribution summed for each word
        words = [self.C.tokenizer.decode(i) for i in conv_b.ids[0]]
        tok_attr = torch.sum((output*vec_dir).squeeze(0), dim=-1)/N
        tok_attr = tok_attr.tolist()
        print(sum(tok_attr))
        return words, tok_attr, pred_class, true_class

    def _line_integral(self, ids, utt_pos, quiet=False):
        #make batches for line integral
        with torch.no_grad():
            embeds = self.model.get_embeds(ids)        #[1,L,d]
            base_embeds = torch.zeros_like(embeds)     #[1,L,d]
            vec_dir = (embeds-base_embeds)

            alphas = torch.arange(1, N+1, device=self.device)/N
            line_path = base_embeds + alphas.view(N,1,1)*vec_dir            
            batches = [line_path[i:i+args.bsz] for i in 
                       range(0, len(line_path), args.bsz)] #[N,L,d]     

        #do line integral through discrete approximation
        output = torch.zeros_like(input_embeds)
        for embed_batch in tqdm(batches, disable=quiet):            
            embed_batch.requires_grad_(True)
            y = self.model({'inputs_embeds':embed_batch}, utt_pos=utt_pos)
            preds = F.softmax(y, dim=-1)[:, pred_idx]
            torch.sum(preds).backward()

            grads = torch.sum(embed_batch.grad, dim=0)
            output += grads.detach().clone()
        
        output = (output*vec_dir).squeeze(0)/N
        
        #sum over all elements of embedding for word
        tok_attr = torch.sum(output, dim=-1)
        
        return tok_attr
