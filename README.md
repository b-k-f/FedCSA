# FedCSA: A Novel Federated Learning Client Selection with Anomaly Detection Approach for IoT Systems

**Author:** Bouchra Fakher  
**Published in:** 2025 WCNC  
**Date:** 2025  
**DOI:** [[DOI link]](https://doi.org/10.1109/WCNC61545.2025.10978709)  

## Abstract

Federated Learning (FL) is emerging as a crucial approach to enhance data privacy and security, particularly in smart buildings and Internet of Things (IoT) ecosystems. By distributing learning across multiple clients, FL minimizes the need for centralized data transfers. This decentralized approach allows clients to collaboratively improve machine learning models without sharing raw data, and only their model updates are sent to a central server for aggregation. However, the problem with the existing aggregation approaches is randomizing and fixing the choice of participating clients during the FL process without evaluating the quality and potential anomalies in individual client model updates during training rounds, which can impact the aggregation and the global model performance. Therefore, we introduce a novel dynamic client selection approach called FedCSA, which selects clients using a scoring mechanism that prioritizes model quality and anomaly detection. Clients with scores above a threshold are chosen, and any client not selected for several consecutive cycles is flagged as malicious and removed. This ensures bad or malfunctioning clients are secluded and not selected during the remaining training rounds. Simulation results using smart building datasets demonstrate superior global per-formance compared to other client selection methods, including loss, SMAPE, RMSE, and MAE, across varying client numbers. This shows the scalability and consistency of our method for large-scale FL tasks with IoT time-series data.

# If you use this work, please cite it as follows:

```bibtex
@inproceedings{fakher2025fedcsa,
  title={FedCSA: A Novel Federated Learning Client Selection with Anomaly Detection Approach for IoT Systems},
  author={Fakher, Bouchra and Bennis, Ismail and Abouaissa, Abdelhafid and others},
  booktitle={2025 IEEE Wireless Communications and Networking Conference (WCNC)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
