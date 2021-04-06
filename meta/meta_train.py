import torch
import torch.optim as optim
import torch.nn.functional as F
import higher


def meta_train(db, net, meta_opt, n_iter_inner_loop, task_bsize, device):

    net.train()

    task_batch = db.task_batch("train", task_bsize)

    inner_opt = optim.SGD(net.parameters(), lr=0.1)

    task_qry_acc = 0
    meta_opt.zero_grad()

    for task in task_batch:
        x_spt, x_qry, y_spt, y_qry = task
        x_spt = x_spt.to(device)
        y_spt = y_spt.to(device, dtype=torch.int64)
        x_qry = x_qry.to(device)
        y_qry = y_qry.to(device, dtype=torch.int64)

        with higher.innerloop_ctx(net, inner_opt, device, copy_initial_weights=False) as (fnet, diffopt):
            for _ in range(n_iter_inner_loop):
                spt_logits = fnet(x_spt)
                spt_loss = F.cross_entropy(spt_logits, y_spt)
                diffopt.step(spt_loss)

            qry_logits = fnet(x_qry)
            qry_loss = F.cross_entropy(qry_logits, y_qry)
            qry_loss.backward()
            
            fnet.eval()
            qry_logits = fnet(x_qry).detach()
            task_qry_acc += (qry_logits.argmax(dim=1) ==
                             y_qry).sum().item()/(y_qry.size()[0]*len(task_batch))

    meta_opt.step()
    return task_qry_acc