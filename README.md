# RealCQA
## Scientific Chart Question Answering as a Test-bed for First-Order Logic

### [Arxiv Link](https://arxiv.org/pdf/2308.01979.pdf)


### Updates - 8/23
#### To Do Checklist

- [] Release Dataset
  - [x] Release data
  - [] Benchmark Matcha
      - [x] Matcha-ChartQA
      - [x] Matcha-PlotQA
      - [] Matcha-FineTune
  - [] Release Premises
      - [] Natural Language Premises
      - [] Abstract Syntax Tree sympy
- [] Evaluation Script
    - [] Add explain and leaderboard
    - [] Upload Output sample
    - [] Visualisation       


#### Run CMD  for Matcha Baseline
---
```
python -m torch.distributed.launch --nproc_per_node=4 code/cqa_hf.py --img_dir /path/to/images --cjson_dir /path/to/chart/jsons --json_dir /path/to/qa/jsons --output /path/to/output
```

#### Dataset Details ----

####  folder structure 
RQA_V0.1.8.23.zip-- [Download Link ](https://drive.google.com/file/d/1QJ1v1x7XRILRjCB2YKJ6QejgdznuLuTY/view?usp=sharing)
---
The zip folder cantains of images, the chartinfo challenge json and the QA Json 

./images       

./json 

./qa

./test_filenames.txt


| TOTAL --                                 |                   |
|------------------------------------------|-------------------|
| images                                   |  28 k             |
| questions unique qa_id                   | 2 mil             |
| Total (Test) Evaluated                   | (charts/qa pairs) |
| Sampling 1 Exhaustive                    | 9,357 / 367,139   |
| Sampling 2 Lower Bound removed           | 9,357 / 322,404   |
| Sampling 3 Lower + Upper Bound removed : | 9,357 / 276,091   |
| Sampling 4 Upper Bound                   | 9,357 / 231,356   |
| Sampling 5 Flat 150                      | 9,357 / 203,735   |


#### Evaluation 

go to ./code/evaluation 

requires gt json and predicted json 

##### ground truth json 
should have structure 

```
[{
    "taxonomy id": "2e", 
    "QID": "3",
    "question": "Where does the legend appear in the chart?",
    "answer": "Upper Right"
    "answer_type": "String", 
    "qa_id": "nIHYqYHRYHET"  }  , 

{} ...
]
```

##### predicted output
should list of jsons per groundtruth images, each json should have at least two keys : "qa_id" : the_unique_qa_id and "predicted_answer" : predicted_output_from_model

```

[
  {
    "qa_id": "JMtriZKutxef",
    "predicted_answer": "Yes"
  },
  {
    "qa_id": "RDImiPEPDGAR",
    "predicted_answer": "2"
  },
  {
    "qa_id": "JJinRauejKgm",
    "predicted_answer": "0.03"
  },
  {
    "qa_id": "yEPDZLACdbwM",
    "predicted_answer": "No"
  },
  {..}
]
```
---


![Screenshot 2023-03-03 131707](https://github.com/cse-ai-lab/RealCQA/assets/6873582/3b8b0728-433d-4798-afb6-2d6cdbbb6541)

Existing datasets in chart visual QA either are fully synthetic charts generated from synthetic data (right sector) or synthetic charts generated from real data (left sector). 

None of these datasets handle the complexity of the distribution of real-world charts found in scientific literature. 

We introduce the first chart QA dataset (RealCQA) in the third category (lower sector) which consists of real-world charts extracted from scientific papers along with various categories of QA pairs.

### Overview 


https://github.com/cse-ai-lab/RealCQA/assets/6873582/60401ee5-1597-4e7b-b120-c1295111ba78


![poster of REAL QA](https://github.com/cse-ai-lab/RealCQA/blob/main/figs/REALcqa_v2.svg "Real CQA")

---

