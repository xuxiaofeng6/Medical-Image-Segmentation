

def get_dataset(args, mode, **kwargs):
    
    if args.dimension == '2d':
        if args.dataset == 'acdc':
            from .dim2.dataset_acdc import CMRDataset

            return CMRDataset(args, mode=mode, k_fold=args.k_fold, k=kwargs['fold_idx'], seed=args.split_seed)

    else:
        if args.dataset == 'ixi':
            from .dim3.dataset_ixi import ixiDataset

            return ixiDataset(args, mode=mode, seed=args.seed)



