class Agent:
    """
    Agent is constituted of Modules and interact with Environment 
    """
    def serialize_modules(self, ckpt_dir):
        for module in self._modules:
            modules.serialize(ckpt_dir)
    
    def act(self):
        raise NotImplementedError




        
    