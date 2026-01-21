# FDA-Opt: Communication-Efficient Federated Learning
Real-system implementation for the [CREXDATA](https://crexdata.eu/) Project of
```bibtex
@misc{theologitis2025communication,
    title={FDA-Opt: Communication-Efficient Federated Fine-Tuning of Language Models},
    author={Michail Theologitis and Vasilis Samoladas and Antonios Deligiannakis},
    year={2025},
    eprint={2505.04535},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
building upon the work of
```bibtex
@inproceedings{theologitis2025fda,
  author       = {Michail Theologitis and
                  Georgios Frangias and
                  Georgios Anestis and
                  Vasilis Samoladas and
                  Antonios Deligiannakis},
  title        = {Communication-Efficient Distributed Deep Learning via Federated Dynamic
                  Averaging},
  booktitle    = {Proceedings 28th International Conference on Extending Database Technology,
                  {EDBT} 2025, Barcelona, Spain, March 25-28, 2025},
  pages        = {411--424},
  publisher    = {OpenProceedings.org},
  year         = {2025},
  url          = {https://doi.org/10.48786/edbt.2025.33},
  doi          = {10.48786/EDBT.2025.33}
}
```

## 🛠️ Architecture 🛠️

We aim to support the execution of multiple federated learning (FL) jobs, each arriving with its own configuration—such as the number of clients, chosen optimizer, model name, and many more—through a Kafka topic. Each FL job is completely isolated and decoupled from others.

To coordinate these jobs, we introduce a dedicated orchestrator, called the *mediator*. The mediator is responsible for managing the full lifecycle of every FL job—before, during, and after execution.
Specifically, the *mediator* process continuously listens to a Kafka topic for incoming JSON messages. Upon receiving one, it:

1. Parses the configuration.

2. Sets up the FL cluster by launching a server and the appropriate number of clients.

3. Ensures that all components are connected using the provided network details (e.g., IPs and ports).

4. Makes sure that the deployed FL cluster is booted up without errors

In summary, the mediator automates the deployment and coordination of each FL job, ensuring reliable and scalable orchestration across multiple jobs running concurrently or sequentially.

![alt text](https://github.com/miketheologitis/FDA-Opt-Sys/blob/main/arch.png?raw=true "Project Architecture")

## 📜 Project Structure 📜

```
FDA-Opt-Sys/
├── README.md                              # Overview & setup instructions
├── mediator.txt                           # The mediator script that continuously runs and deploys clusters for FL
├── evaluate_model.py                      # Evaluation script for the models saved in the checkpoint folder
│
├── fdaopt/                                # Core implementation
│   ├── __init__.py
│   ├── client.py                          # The FL client script
│   ├── data.py                            # Data processing with HuggingFace
│   ├── fda_strategies.py                  # The FedOpt strategies augmented with our FDA
│   ├── networking.py                      # Helpful networking ops for inner-cluster communication (e.g., sockets, etc.)
│   ├── server.py                          # The FL server script
│   ├── sketch.py                          # The AMS-Sketch for FDA
│   └── training.py                        # Training functions with or without FDA
│
├── data/                                  # Data partitioning stuff
│   ├── save_data.py                       # Script for partitioning some HuggingFace dataset and saving it
│   └── /glue/mrpc/*                       # MRPC dataset partitioned in various ways with `save_data.py` (e.g., between 10 clients or 2 clients)
│
├── hyperparameters/*                      # Hyperparameters in JSON format
│
└── logs/                                  # Logs for all running entities across different jobs (mediator, servers, clients)
    ├── *.log                              # Logs
    └── checkpoints/                       # Model Checkpoints saved after training
         └── <model‑name>-<job-id>.pth     # e.g., prajjwal1-bert-tiny-039f3834.pth
```


## ⚙️ Install dependencies and project ⚙️

```bash
pip install flwr torch transformers datasets confluent-kafka
```


## 🚀 Run 🚀

To launch a federated learning job, all you need to do is start the mediator. 
Once it's running, it will continuously listen to a Kafka topic for incoming FL jobs 
 (in JSON format) and automatically handle their deployment.

Each job configuration includes everything needed to spin up a 
complete FL cluster (server and clients). The mediator ensures the entire 
process—from setup to teardown—is fully automated.

In the following example, we'll demonstrate the two steps:

1. Start the mediator process to listen for FL jobs.
2. Send FL jobs as JSON configurations to the Kafka topic.

### 1. Start the Mediator:

We start the mediator entity and let it run in the background:

```bash
python mediator.py --cleanup
```

Note: the `--cleanup` command instructs the mediator to clean-up logs after it is destroyed/killed by the user.

### 2. Send JSON files to Kafka

Go to `/hyperparameters` and send the `test_1-minified.json` to the `FedL` topic:
```bash
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic  FedL < test_1-minified.json
```
Then, send another JSON, `test_2-minified.json`, with different hyperparameters (the mediator continues to listen):
```bash
kafka-console-producer.sh --bootstrap-server localhost:9092 --topic  FedL < test_2-minified.json
```
And so on...

Note: we expect a minified JSON (no whitespaces, tabs, newlines, comments, etc.); hence the name!

## 📰 Read the Logs 📰

Go to `/logs` and monitor the two different jobs we submitted. You can read anything you like: the mediator's logs,
one of the two FL server's logs, or any of the client's logs!

## 🔬 Testing Purposes 🔬
We can also start the mediator locally for testing purposes. Run 1-time job:
```bash
python mediator.py --cleanup --local <path_to_folder>/test_1-minified.json
```

## 🥂 Evaluate 🥂

After each FL Job finishes we save the final model at `logs/checkpoints/<model‑name>-<job-id>.pth`. We can evaluate it as follows:

```bash
python evaluate_model.py \
  --model_checkpoint prajjwal1/bert-tiny \
  --local_weights logs/checkpoints/prajjwal1-bert-tiny-039f3834.pth \
  --ds_path glue \
  --ds_name mrpc \
  --device cuda:1 \
  --num_labels 2
```

## 📋 Different Options - JSON Schema 📋

Every federated learning (FL) job in our system is defined by a JSON configuration. 
These configurations give us fine-grained control over the entire 
FL process—from the model and dataset to the training strategy and network setup.
Each JSON payload is pushed through Kafka and triggers a fully independent FL cluster.

Given the flexibility of the system, there are many possible combinations of parameters we can provide. 
While it's not
feasible to document every combination, we provide here a structured schema of the supported fields, 
along with explanations and examples of how to use them.

Some fields are optional, 
and our system is designed to adapt 
dynamically based on what you provide. 
For example, if you're using the FDA strategy and omit the theta parameter, 
the algorithm will dynamically adjust it during training—see our [FDA-Opt](https://arxiv.org/abs/2505.04535) paper for more details on this behavior.

The sections below break down the configuration fields into logical groups 
(model, dataset, training, server, clients) so we can easily craft powerful job 
descriptions that meet any needs.

---
- `model`:
  - `checkpoint`: HuggingFace model checkpoint: Specifies the pre-trained model to use (e.g., "roberta-base").
  - `num_labels`: Defines the number of output labels for the task.

- `dataset`:
  - `path`: HuggingFace path or name of the dataset (e.g., "glue"). See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset).
  - `batch_size`: Batch size for data processing.
  - `dirichlet_alpha`: Dirichlet data distribution parameter for non-IID partitioning.
  - `name`: HuggingFace name of the dataset. See [here](https://huggingface.co/docs/datasets/v3.2.0/en/package_reference/loading_methods#datasets.load_dataset). (**Optional**)

- `training`:
  - `num_clients`: Total number of clients in the federation.
  - `clients_per_round`: Number of clients participating in each training round.
  - `local_epochs`: Number of epochs each client trains locally.
  - `total_rounds`: Total number of federated training rounds.

- `server`:
    - `network`:
      - `ip`: Server IP address for Flower.
      - `port`: Server port number for Flower.
      - `ip_pull_socket`: Server IP address for PULL socket used in FDA.
      - `port_pull_socket`: Server IP address for PULL socket used in FDA.
    - `strategy`: Specifies the federated learning strategy used in Flower.
      - `name`: Name of the strategy (e.g., `"FedAdam"`). See [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html).
      - `fda`: Whether or not to use FDA extention. Either `True` or `False`
      - `theta`: FDA-Specific and also **Optional** (if `fda` is `True` and this *empty* then we change the value dynamically—see [paper](https://arxiv.org/abs/2505.04535))
      - **Optional** (All strategy-specific applicable hyperparameters from [here](https://flower.ai/docs/framework/ref-api/flwr.server.strategy.html)):
         - e.g., `eta`: Server-side learning rate hyperparameter.

- `clients`:
  - `network`: **List** of client details:
    - `id`: Unique identifier for each client.
    - `ip`: IP address for client PULL socket.
    - `port`: Port number for client PULL socket.
    - `data_path`: Local dataset path for each client. **Optional** (when *empty* the client used HuggingFace)
  - `lr`: Learning rate for the client-side optimizer (SGD). The optimizer is fixed as Stochastic Gradient Descent (SGD) for all clients.

---

**Note**: We expect a minified JSON 
(no whitespaces, tabs, newlines, comments, etc.)

---

## License 

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

