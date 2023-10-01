import sys
import time

import torch

from .utils import Meter, TextArea
try:
    from .datasets import CocoEvaluator, prepare_for_coco
except:
    pass


def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    for p in optimizer.param_groups:
        p["lr"] = args.lr_epoch

    iters = len(data_loader) if args.iters < 0 else args.iters
    print('iters', iters)

    t_m = Meter("total")
    m_m = Meter("model")
    b_m = Meter("backward")
    model.train()
    A = time.time()
    for i in range(0, iters, args.timestep):
        image, target = data_loader[i]
        T = time.time()
        num_iters = epoch * len(data_loader) + i
        if num_iters <= args.warmup_iters:
            r = num_iters / args.warmup_iters
            for j, p in enumerate(optimizer.param_groups):
                p["lr"] = r * args.lr_epoch
                   
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items() if v is not None}
        target = [target]
        for j in range(i+1, i+args.timestep):
            _, tmp = data_loader[j]
            tmp = {k: v.to(device) for k, v in tmp.items() if v is not None}
            target.append(tmp)
        S = time.time()
        
        losses = model(image, target=target)
        total_loss = sum(losses.values())
        m_m.update(time.time() - S)
            
        S = time.time()
        total_loss.backward()
        b_m.update(time.time() - S)
        
        optimizer.step()
        optimizer.zero_grad()

        if num_iters % args.print_freq == 0:
            print("{}\t".format(num_iters), "\t".join("{:.3f}".format(l.item()) for l in losses.values()))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
           
    A = time.time() - A
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}, backward: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg,1000*b_m.avg))
    return A / iters
            

def evaluate(model, data_loader, device, args, generate=True):
    iter_eval = None
    if generate:
        iter_eval = generate_results(model, data_loader, device, args)

    dataset = data_loader #
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(dataset.coco, iou_types)

    results = torch.load(args.results, map_location="cpu")
    #print('coco results', results)

    S = time.time()
    coco_evaluator.accumulate(results)
    print("accumulate: {:.1f}s".format(time.time() - S))

    # collect outputs of buildin function print
    temp = sys.stdout
    sys.stdout = TextArea()

    coco_evaluator.summarize()

    output = sys.stdout
    sys.stdout = temp
        
    return output, iter_eval
    
    
# generate results file   
@torch.no_grad()   
def generate_results(model, data_loader, device, args):
    iters = len(data_loader) if args.iters < 0 else args.iters
        
    t_m = Meter("total")
    m_m = Meter("model")
    coco_results = []
    model.eval()
    A = time.time()
    for i in range(0, len(data_loader), args.timestep):
        image, target = data_loader[i]
        T = time.time()
        
        image = image.to(device)
        target = {k: v.to(device) for k, v in target.items() if v is not None}

        S = time.time()
        #torch.cuda.synchronize()
        # t = list(range(1, args.timestep))
        # t = list(range(2, args.timestep, 2))
        t = [args.timestep-1]
        output = model(image, time=t)

        m_m.update(time.time() - S)

        for t in range(1, args.timestep, 1):
            image_t, target_t = data_loader[i+t]
            image_t = image_t.to(device)
            target_t = {k: v.to(device) for k, v in target_t.items() if v is not None}
            if t == 1:
                res = output[0]
                res["future_boxes"] = res["boxes"]
        
                prediction = {target["image_id"].item(): {k: v.cpu() for k, v in res.items()}}
                coco_results.extend(prepare_for_coco(prediction))        
                prediction = {target_t["image_id"].item(): {k: v.cpu() for k, v in output[0].items()}}
                coco_results.extend(prepare_for_coco(prediction))
            else:
                # prediction = {target_t["image_id"].item(): {k: v.cpu() for k, v in output[(t//2-1)].items()}}
                prediction = {target_t["image_id"].item(): {k: v.cpu() for k, v in output[(t//1-1)].items()}}
                coco_results.extend(prepare_for_coco(prediction))
        # print('coco_results length:', len(coco_results))


            # output[t-1]["future_boxes"] = output[t-1]["boxes"]
            # if t == 1:
        
            #     prediction = {target["image_id"].item(): {k: v.cpu() for k, v in output[0].items()}}
            #     coco_results.extend(prepare_for_coco(prediction))        
            #     prediction = {target_t["image_id"].item(): {k: v.cpu() for k, v in output[0].items()}}
            #     coco_results.extend(prepare_for_coco(prediction))
            # else:
            #     prediction = {target_t["image_id"].item(): {k: v.cpu() for k, v in output[t-1].items()}}
            #     coco_results.extend(prepare_for_coco(prediction))

        t_m.update(time.time() - T)
        if i >= iters - 1:
            break
     
    A = time.time() - A 
    print("iter: {:.1f}, total: {:.1f}, model: {:.1f}".format(1000*A/iters,1000*t_m.avg,1000*m_m.avg))
    torch.save(coco_results, args.results)
        
    return A / iters
    

