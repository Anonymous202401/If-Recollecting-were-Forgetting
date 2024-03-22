import torch
import time
import joblib
from models.Nets import MLP, CNNMnist, CNNCifar,Logistic,LeNet,resnet18
import torch
import os
import torch
from torchvision.models import resnet18


def getapproximator_celeba(args,img_size,Dataset2recollect,indices_to_unlearn):
    torch.cuda.empty_cache()
    net_t = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_t = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_t = CNNMnist(args=args).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'fashion-mnist':
        net_t = LeNet().to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'celeba':
        net_t = resnet18(num_classes=2).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net = resnet18(pretrained=True).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_t = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    elif args.model == 'logistic':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_t = Logistic(dim_in=len_in,  dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    net_t.eval()    
    print(net_t)
    # net_t setup
    lr=args.lr
    loss_func = torch.nn.CrossEntropyLoss()
    approximator = {i: [torch.zeros_like(param) for param in net_t.parameters()] for i in indices_to_unlearn}
    for iter in range(args.epochs):
        # load file
        path1 = "./Checkpoint/Resnet/model_{}_checkpoints". format(args.model)
        file_name = "check_{}_remove_{}_{}_seed{}_iter_{}.dat". format(args.dataset, args.num_forget,args.epochs,args.seed,iter)
        file_path = os.path.join(path1, file_name)
        info = joblib.load(file_path)  
        dataset = Dataset2recollect

        # influence
        for t in range(len(info)):
            net_t.zero_grad()
            model_t, batch_idx = info[t]["model_list"],info[t]["batch_idx_list"]
            if args.model==resnet18:
                net_t.train()
            else:  net_t.eval()
            net_t.load_state_dict(model_t)
            batch_images_t, batch_labels_t = [], []
            loss_batch = 0.0
            for i in batch_idx:
                image_i, label_i, index_i = dataset[i]
                image_i ,label_i= image_i.unsqueeze(0).to(args.device), torch.tensor([label_i]).to(args.device)
                batch_images_t.append(image_i)
                batch_labels_t.append(label_i)        
                if i in indices_to_unlearn:
                    log_probs = net_t(image_i)
                    loss_i = loss_func(log_probs , label_i)
                    net_t.zero_grad()
                    for param in net_t.parameters():
                        loss_i += 0.5 * args.regularization * (param * param).sum()
                    loss_i.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=net_t.parameters(), max_norm=args.clip, norm_type=2)
                    for j, param in enumerate(net_t.parameters()):
                        approximator[i][j] += (param.grad.data * lr*(args.lr_decay**(len(info)*iter+t))) / len(batch_idx)
            # print(net_t.state_dict())
            log_probs=0
            loss_batch =0 
            grad_norm = 0
            batch_images_t = torch.cat(batch_images_t, dim=0)
            batch_labels_t = torch.cat(batch_labels_t, dim=0)
            log_probs = net_t(batch_images_t)
            loss_batch = loss_func(log_probs, batch_labels_t)    
            print("Recollecting Model  {:3d}, Training Loss: {:.2f}".format(t,loss_batch))
            for param in net_t.parameters():
                loss_batch += 0.5 * args.regularization * (param * param).sum()
            grad_params = torch.autograd.grad(loss_batch, net_t.parameters(), create_graph=True, retain_graph=True)
            grad_norm = torch.norm(torch.cat([grad.view(-1) for grad in grad_params]))
            if grad_norm > args.clip:
                scaling_factor = args.clip / grad_norm
                grad_params = [grad * scaling_factor for grad in grad_params]
            t_start = time.time()   
            for i in indices_to_unlearn: 
                net_t.zero_grad()
                HVP_i=torch.autograd.grad(grad_params,net_t.parameters(),approximator[i],retain_graph=True)
                for j, param in enumerate(net_t.parameters()):
                    approximator[i][j]=approximator[i][j] - (lr* (args.lr_decay**(len(info)*iter+t)) * HVP_i[j].detach())
            del HVP_i,loss_batch,grad_params
            t_end = time.time()
            print("Computaion For step {} Time Elapsed:  {:.2f}s \n".format(iter,t_end - t_start))    

    return approximator
    

