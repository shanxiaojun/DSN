# Enhancing Knowledge Transfer for Task Incremental  Learning with Data-free Subnetwork

## Abstract

As there exist competitive subnetworks within a dense network in concert with *Lottery Ticket Hypothesis*, we introduce a novel neuron-wise task incremental learning method, namely *Data-free Subnetworks (DSN)*, which attempts to enhance the elastic knowledge transfer across the tasks that sequentially arrive. Specifically, DSN primarily seeks to transfer knowledge to the new coming task from the learned tasks by selecting the affiliated weights of a small set of neurons to be activated, including the reused neurons from prior tasks via neuron-wise masks. And it also transfers possibly valuable knowledge to the earlier tasks via data-free replay. Especially, DSN inherently relieves the catastrophic forgetting and the unavailability of past data or possible privacy concerns. The comprehensive experiments conducted on four benchmark datasets demonstrate the effectiveness of the proposed DSN in the context of task-incremental learning by comparing it to several state-of-the-art baselines. In particular, DSN enables the knowledge transfer to the earlier tasks, which is often overlooked by prior efforts.

## Usage

1. Create a python 3 conda environment (check the requirements.txt file)

2. The following folder structure is expected at runtime. From the  folder:
   
   - src/ : Where all the scripts lie (already produced by the repo). Include dataloader, network, and approach.
   - dat/ : Place to put/download all data sets

3. The main script is src/newrun.py. To run the experiment, use 
   
   `python src/newrun.py`
