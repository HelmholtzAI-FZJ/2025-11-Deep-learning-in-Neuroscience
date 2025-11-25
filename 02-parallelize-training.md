---
author: Alexandre Strube // Sabrina Benassou // Anoop K. Chandran 
title: Deep Learning for Neuroscience
# subtitle: A primer in supercomputers`
date: November 28, 2025

---

## Before Starting

- We need to download some code

    ```bash
    cd $HOME/course
    git clone https://github.com/HelmholtzAI-FZJ/2025-11-Deep-learning-in-Neuroscience.git
    ```

- Move to the correct folder
    
    ```bash
    cd 2025-11-Deep-learning-in-Neuroscience/code/parallelize/
    ```

---

## What this code does 

- It trains a [Transformer](https://arxiv.org/pdf/1706.03762) language model on [WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2) dataset to predict the next word in a sequence. 
- **Transformers** is a deep learning model architecture that uses self-attention to process sequences in parallel.
- **WikiText-2** is a word-level language modeling dataset consisting of over 2 million tokens extracted from high-quality Wikipedia articles. 

---

## What this code does 

- Again, this is not a deep learning course.
- If you are not familiar with the model and the dataset, just imagine it as a black box: you provide it with text, and it generates another text.

    ![](images/black_box.svg)

---

## Libraries

- You already downloaded the libraries yesterday that we will use today:
    - **PyTorch:** A deep learning framework for building and training models.
    - **datasets** A library from Hugging Face used to access, load, process, and share large datasets.
    
---

Let's have a look at the files **```to_distrbuted_training.py```** and **```run_to_distributed_training.sbatch```** in the repo.

![](images/look.jpg)

---

## Run the Training Script

- There are TODOs in these two files. **Do not modify the TODOs for now**. The code is already working, so you don’t need to make any changes at this point.
- Now run:

    ```bash
    sbatch run_to_distributed_training.sbatch 
    ```

- Spoiler alert 🚨

- The code won't work.

- Check the output and error files

---

## What is the problem?

- Remember, there is no internet on the compute node.
- Therefore, you should:
    - **Comment out** lines 76 **to** 141.
    - Activate your environment:

        ```bash
        source $HOME/course/sc_venv_template/activate.sh
        ```

    - Run:

        ```bash
        python to_distributed_training.py
        ```

    - **Uncomment back** lines 76-141.
    - Finally, run your job again 🚀:

        ```bash
        sbatch run_to_distributed_training.sbatch
        ```

---

## JOB Running

- Congrats, you are training a DL model on the supercomputer using one GPU 🎉

--- 

## llview

- You can monitor your training using [llview](https://go.fzj.de/llview-jureca). 
- Use your Judoor credentials to connect.
- Check the job number that you are intrested in.
    ![](images/llview_job.png){height=400px}

---


## llview

- Go to the right to open the PDF document. **It may take some time to load the job information, so please wait until the icon turns blue**.
    ![](images/llview.png){height=450px}

---

## llview

- You have many information about your job once you open the PDF file.
    ![](images/llview_info.png)

---

## GPU utilization 

- You can see that in fact we are using **1 GPU**

    ![](images/llview_gpu_1.png)

---

## GPU utilization   

- It is a waste of resources.

- The training takes time (1h32m according to llview).

- Then, can we run our model on multiple GPUs ?

---

## What if

- At line 3 in file **```run_to_distributed_training.sbatch```**, we increase the number of GPUs to 4:

    ```bash
    #SBATCH --gres=gpu:4
    ```

- And run our job again

    ```bash
    sbatch run_to_distributed_training.sbatch
    ```

--- 


## llview

- We are still using **1 GPU**

- ![](images/llview_gpu_2.png)

---

## We need communication

- Without correct setup, the GPUs might not be utilized.

- Furthermore, we don't have an established communication between the GPUs

    ![](images/dist/no_comm.svg){height=400px}

---

## We need communication

![](images/dist/comm1.svg){height=500px}

---

## We need communication

![](images/dist/comm2.svg){height=500px}

---

## collective operations

- The GPUs use collective operations to communicate and share data in parallel computing
- The most common collective operations are: All Reduce, All Gather, and Reduce Scatter

---

## All Reduce 

![](images/dist/all_reduce.svg)

- Other operations, such as **min**, **max**, and **avg**, can also be performed using All-Reduce.

---

## All Gather

![](images/dist/all_gather.svg)

--- 

## Reduce Scatter

![](images/dist/reduce_scatter.svg)

--- 

## Terminologies

- Before going further, we need to learn some terminologies

---

## World Size

![](images/dist/gpus.svg){height=550px}

---

## Rank

![](images/dist/rank.svg){height=550px}

---

## local_rank

![](images/dist/local_rank.svg){height=550px}

---

## Now

That we have understood how the devices communicate and the terminologies used in parallel computing, 
we can move on to distributed training (training on multiple GPUs).

---

## Distributed Training

- Parallelize the training across multiple nodes, 
- Significantly enhancing training speed and model accuracy.
- It is particularly beneficial for large models and computationally intensive tasks, such as deep learning.[[1]](https://pytorch.org/tutorials/distributed/home.html)


---

## Distributed Data Parallel (DDP)

[DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is a method in parallel computing used to train deep learning models across multiple GPUs or nodes efficiently.

![](images/ddp/ddp-2.svg){height=400px}

--- 

## DDP

![](images/ddp/ddp-3.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-4.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-5.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-6.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-7.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-8.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-9.svg){height=500px}

--- 

## DDP

If you're scaling DDP to use multiple nodes, the underlying principle remains the same as single-node multi-GPU training.

---

## DDP

![](images/ddp/multi_node.svg){height=500px}

---

## DDP recap

- Each GPU on each node gets its own process.
- Each GPU has a copy of the model.
- Each GPU has visibility into a subset of the overall dataset and will only see that subset.
- Each process performs a full forward and backward pass in parallel and calculates its gradients.
- The gradients are synchronized and averaged across all processes.
- Each process updates its optimizer.

---

## Let's start coding!

- Whenever you see **TODOs**💻📝, follow the instructions to either copy-paste the code at the specified line numbers or type it yourself.

- Depending on how you copy and paste, the line numbers may vary, but always refer to the TODO numbers in the code and slides.

---

## Setup communication

- We need to setup a communication among the GPUs. 
- For that we would need the file **```distributed_utils.py```**.
- **TODOs**💻📝:
    1. Import **```distributed_utils```** file at line 11:
        
        ```python 
        # This file contains utility_functions for distributed training.
        from distributed_utils import *
        ```
    2. Then **remove** lines 65 and 66:

        ```python
        ## TODO 2-3: Remove this line and replace it with a call to the utility function setup().
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ```
    3. and **add** at line 65 a call to the method **```setup()```** defined in **```distributed_utils.py```**: 

        ```python
        # Initialize a communication group and return the right identifiers.
        local_rank, rank, device, world_size = setup()
        ```

---

## Setup communication

What is in the **```setup()```** method ?

```python
def setup():
    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')
    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))
    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])
    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)
    # Different random seed for each process.
    torch.random.manual_seed(1000 + torch.distributed.get_rank())

    return local_rank, rank, device
```

---

## DistributedSampler 

- **TODO 4**💻📝:

    - At line 76, instantiate a **DistributedSampler** object for each set to ensure that each process gets a different subset of the data.
    
        ```python
        # DistributedSampler object for each set to ensure that each process gets a different subset of the data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                        shuffle=True, 
                                                                        seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        ```

---

## DataLoader

- **TODO 5**💻📝:

    - At line 85, **REMOVE** **```shuffle=True```** in the DataLoader of train_loader and **REPLACE** it by **```sampler=train_sampler```**
        
        ```python 
        train_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                sampler=train_sampler, # pass the sampler argument to the DataLoader
                                num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')),
                                pin_memory=True)
        ```

---

## DataLoader

- **TODO 6**💻📝:

    -  At line 90, pass **val_sampler** to the sampler argument of the val_dataLoader

        ```python
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler, # pass the sampler argument to the DataLoader
                                pin_memory=True)
        ```

- **TODO 7**💻📝:

    - At line 94, pass **test_sampler** to the sampler argument of the test_dataLoader

        ```python
        test_loader = DataLoader(test_dataset,
                                batch_size=args.test_batch_size,
                                sampler=test_sampler, # pass the sampler argument to the DataLoader
                                pin_memory=True)    
        ```

--- 

## Model

- **TODO 8**💻📝:

    - At line 110, wrap the model in a **DistributedDataParallel** (DDP) module to parallelize the training across multiple GPUs.
    
        ```python 
        # Wrap the model in DistributedDataParallel module 
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
        )
        ```

---

## Sampler 

- **TODO 9**💻📝:

    - At line 124, **set** the current epoch for the dataset sampler to ensure proper data shuffling in each epoch

        ```python
        # Pass the current epoch to the sampler to ensure proper data shuffling in each epoch
        train_sampler.set_epoch(epoch)
        ```

---

## All Reduce Operation

- **TODO 10**💻📝:

    - At **lines 37 and 58**, Obtain the global average loss across the GPUs.

        ```python
        # Return the global average loss.
        torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
        ```

---

## print

- **TODO 11**💻📝:

    - **Replace** all the ```print``` methods by **```print0```** method defined in **```distributed_utils.py```** to allow only rank 0 to print in the output file.
    
    - At **line 130** 

        ```python
        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
        ```

    - At **line 142**
    
        ```python
        # We use the utility function print0 to print messages only from rank 0.
        print0('Final test loss:', test_loss.item())
        ```

---

## print

The definition of the function **print0** is in **```distributed_utils.py```**

```python
functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)
```

---

## Save model 

- **TODO 12**💻📝:

    - At **lines 137 and 146**, replace torch.save method with the utility function save0 to allow only the process with rank 0 to save the model.
 
        ```python 
        # We allow only rank=0 to save the model
        save0(model, 'model-best.pt')
        ```
        ```python 
        # We allow only rank=0 to save the model
        save0(model, 'model-final.pt')
        ```

---

## Save model 

The method **save0** is defined in **```distributed_utils.py```**

```python
functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)
```

--- 

## Destroy Process Group

- **TODO 13**💻📝:

    - At **line 149**, destroy every process group and backend by calling destroy_process_group() 

        ```python 
        # Destroy the process group to clean up resources
        destroy_process_group()
        ```

---

## Destroy Process Group

The method **destroy_process_group** is defined in **```distributed_utils.py```**

```python
def destroy_process_group():
    """Destroy the process group."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
```

---

## We are almost there

- That's it for the **to_distributed_training.py** file. 
- But before launching our job, we need to add some lines to **run_to_distributed_training.sbatch** file 

---

## Setup communication

In **```run_to_distributed_training.sbatch```** file:

- **TODOs 14**💻📝: 
    - At line 3, increase the number of GPUs to 4 if it is not already done.

        ```bash
        #SBATCH --gres=gpu:4
        ```

    - At line 19, pass the correct number of devices.

        ```bash
        # Set up four visible GPUs that the job can use 
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        ```

---

## Setup communication

Stay in **```run_to_distributed_training.sbatch```** file:

- **TODO 15**💻📝: we need to setup **MASTER_ADDR** and **MASTER_PORT** to allow communication over the system.

    - At line 22, add the following:

        ```bash
        # Extracts the first hostname from the list of allocated nodes to use as the master address.
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
        # Modifies the master address to allow communication over InfiniBand cells.
        MASTER_ADDR="${MASTER_ADDR}i"
        # Get IP for hostname.
        export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
        export MASTER_PORT=7010
        ```

---

## Setup communication

We are not done yet with **```run_to_distributed_training.sbatch```** file:

- **TODO 16**💻📝: 
    
    - We **remove** the lauching script at line 41:
    
        ```bash
        srun --cpu_bind=none python to_distributed_training.py 
        ```
    
    - We use **torchrun_jsc** instead and pass the following argument: 

        ```bash
        # Launch a distributed training job across multiple nodes and GPUs
        srun --cpu_bind=none bash -c "torchrun_jsc \
            --nnodes=$SLURM_NNODES \
            --rdzv_backend c10d \
            --nproc_per_node=gpu \
            --rdzv_id $RANDOM \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
            to_distributed_training.py "
        ```

---

## Setup communication

- The arguments that we pass are:

    1. **```nnodes=$SLURM_NNODES```**: the number of nodes
    2. **```rdzv_backend c10d```**: the c10d method for coordinating the setup of communication among distributed processes.
    3. **```nproc_per_node=gpu```** the number of GPUs
    4. **```rdzv_id $RANDOM```** a random id which that acts as a central point for initializing and coordinating the communication among different nodes participating in the distributed training. 
    5. **```rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT```** the IP that we setup in the previous slide to ensure all nodes know where to connect to start the training session.
    6. **```rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi)```** The rendezvous host which is responsible for coordinating the initial setup of communication among the nodes.

---

## done ✅

- You can finally run:

    ```bash
    sbatch run_to_distributed_training.sbatch
    ```

---

## llview

- Let's have a look at our job using [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using all the GPUs of the node

- ![](images/llview_gpu_4.png)

--- 

## llview

- And that our job took less time to finish training (25m vs 1h32m with one GPU)

- But what about using more nodes ?

---

## What about using more nodes ?

---

## Multi-node training

- **TODO 17**💻📝: in **```run_to_distributed_training.sbatch```** at line 2, you can increase the number of nodes to 2:

    ```bash
    #SBATCH --nodes=2
    ```

- Hence, you will use 8 GPUs for training.

- Run again:

    ```bash
    sbatch run_to_distributed_training.sbatch
    ```

--- 

## llview

- Open [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using 2 nodes and 8 GPUs.

- ![](images/llview_gpu_8.png)

- And the training took less time (14m)

---

## Amazing ✨

---

## PART 2 RECAP 

- You know where to store your code and your data. 🗂️📄
- You know what distributed training is. 🧑‍💻
- You can submit training jobs on a single GPU, multiple GPUs, or across multiple nodes. 🎮💻
- You are familiar with DDP. ⚙️💡
- You know how to monitor your training using llview. 📊👀

---

## Find Out More

- Here are some useful:

    - Papers:
        - [FSDP paper](https://arxiv.org/pdf/2304.11277)
        - [Pipeline Parallelism](https://arxiv.org/pdf/1811.06965)
        - [Tensor Parallelism](https://arxiv.org/pdf/1909.08053)

    - Tutorials:
        - [PyTorch at JSC](https://sdlaml.pages.jsc.fz-juelich.de/ai/recipes/pytorch_at_jsc/)
        - [PyTorch tutorials GitHub](https://github.com/pytorch/tutorials/tree/main)
        - [PyTorch documentation](https://pytorch.org/tutorials/distributed/home.html)

    - Links
        - [AI Landing Page](https://sdlaml.pages.jsc.fz-juelich.de/ai/)
        - [Other courses at JSC](https://www.fz-juelich.de/en/ias/jsc/education/training-courses)


---

## ANY QUESTIONS??

#### Feedback is more than welcome!

#### Link to [other courses at JSC](https://www.fz-juelich.de/en/jsc/news/events/training-courses)

---

