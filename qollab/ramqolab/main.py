import hybrid_QML3class_4qubit_2depth_model
import hybrid_QML3class_4qubit_4depth_model
import hybrid_QML3class_4qubit_8depth_model

import hybrid_QML3class_8qubit_1depth_model
import hybrid_QML3class_8qubit_2depth_model
import hybrid_QML3class_8qubit_4depth_model

import hybrid_QML3class_16qubit_1depth_model

def main(shots: int = 4096,
    excludeLowProbabilityValues: bool = True,
    lowProbabilityThreshold: float = 0.05):

    
    print("\n====================================")
    print("SUMMARY")
    print("====================================")
    shots = 1024
    results = [hybrid_QML3class_4qubit_2depth_model.main(backend,shots=shots),
               hybrid_QML3class_4qubit_4depth_model.main(backend,shots=shots),
               hybrid_QML3class_4qubit_8depth_model.main(backend,shots=shots),
               hybrid_QML3class_8qubit_1depth_model.main(backend,shots=shots),
               hybrid_QML3class_8qubit_2depth_model.main(backend,shots=shots),
               hybrid_QML3class_8qubit_4depth_model.main(backend,shots=shots),
               hybrid_QML3class_16qubit_1depth_model.main(backend,shots=shots)
               ]

    print("\n====================================")
    print("SUMMARY")
    print("====================================")
    for res in results:
      print("\n=================================")
      print(res["model_name"])
      print(f"Total Molecules : {res['total']}")
      print(f"Correct         : {res['correct']}")
      print(f"Incorrect       : {res['incorrect']}")
      print(f"Test Accuracy   : {res['accuracy']:.4f}")
      print(f"Test Accuracy % : {res['accuracy_pct']:.2f}%")
      print("=================================")