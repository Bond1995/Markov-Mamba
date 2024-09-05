from .mamba_llm import MambaLMHeadModel

def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = MambaLMHeadModel(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")