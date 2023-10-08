from transformers import AutoProcessor, Pix2StructForConditionalGeneration ;
# import requests ;
# from PIL import Image
from dataset.rqa_dataset import RQADataset
from torch.utils.data import DataLoader
from collections import defaultdict
import os 
import argparse
import tqdm 
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser(description="RQA")
parser.add_argument("-img_dir", "--img_dir", help="Path to the images", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/all_test_train_22/images')
parser.add_argument("-cjson_dir", "--cjson_dir", help="Path to chart jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/all_test_train_22/jsons')
parser.add_argument("-json_dir", "--json_dir", help="Path to the qa jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/combined')
parser.add_argument("-batch_size", "--batch_size", default=18)
parser.add_argument("-output", "--output", help="Path to the output file", default='./outputis/')
parser.add_argument("-filter_list", "--filter_list", help="Filter File list for Train/Test", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/test_filenames.txt')

# parser.add_argument("-img_dir", "--image", help="Path to the images", default='/data_local/ICPR2022_CHARTINFO_UB_PMC_TRAIN_v1.0/images/flatall4')
# parser.add_argument("-json_dir", "--json", help="Path to the qa jsons", default='/home/csgrad/sahmed9/reps/RealCQA/code/data/raw/flat')
# parser.add_argument("-o", "--output", help="Path to the output file")
# parser.add_argument("-o", "--output", help="Path to the output file")

args = parser.parse_args()
'''
google/matcha: the base MatCha model, used to fine-tune MatCha on downstream tasks
google/matcha-chartqa: MatCha model fine-tuned on ChartQA dataset. It can be used to answer questions about charts.
google/matcha-plotqa-v1: MatCha model fine-tuned on PlotQA dataset. It can be used to answer questions about plots.
google/matcha-plotqa-v2: MatCha model fine-tuned on PlotQA dataset. It can be used to answer questions about plots.
google/matcha-chart2text-statista: MatCha model fine-tuned on Statista dataset.
google/matcha-chart2text-pew: MatCha model fine-tuned on Pew dataset.


m_ = ['google/matcha-base', 'google/matcha-chartqa', 'google/matcha-plotqa-v1', 'google/matcha-plotqa-v2']
model = Pix2StructForConditionalGeneration.from_pretrained("google/matcha-chartqa").to(0) ; processor = AutoProcessor.from_pretrained("google/matcha-chartqa")
url = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/20294671002019.png"; image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, text="Is the sum of all 4 places greater than Laos?", return_tensors="pt").to(0)
predictions = model.generate(**inputs, max_new_tokens=512)
print(processor.decode(predictions[0], skip_special_tokens=True))
'''
def train():
    # data_dir = args.im
    print('Start Train')
    print('ARGSPACE') 
    print(args)
    # m_ = ['google/matcha-base', 'google/matcha-plotqa-v2', 'matcha-chart2text-statista']
    m_ = ['google/matcha-plotqa-v2']

    cuda_=4
    rqa_dataset = RQADataset(args, transform=None)
    for model_name in m_ : 
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(cuda_) ; 
        model = DistributedDataParallel(model)
        processor = AutoProcessor.from_pretrained(model_name)
        print('Starting MODEL :: ', model_name)
        output_dir = os.path.join(args.output, model_name)
        os.makedirs(output_dir, exist_ok=True)
        predicted_answers = defaultdict(list)
        data_loader = DataLoader(rqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = rqa_dataset.custom_collate)
        for data_batch in tqdm.tqdm(data_loader) :
            img, q, a, qa_id, pmc_id = data_batch['i'],  data_batch['q'],  data_batch['a'] ,  data_batch['qa_'] ,  data_batch['p_']  
            # print('img', img.shape)
            # print('q', [q_.shape for q_ in q])
            # print('a', [q_.shape for q_ in a])

            inputs = processor(images=img, text=q, return_tensors="pt").to(cuda_)
            predictions = model.generate(**inputs, max_new_tokens=512)
            # print('\npred_answer::', pred_answer)
            # print('pmc_id', pmc_id) 
            # print('qa_id', qa_id) 
            for i in range(len(pmc_id)):
                output_filename = os.path.join(output_dir, f"{pmc_id[i]}.json")
                predicted_answer = {
                    "qa_id": qa_id[i],
                    "predicted_answer": processor.decode(predictions[i], skip_special_tokens=True) ,
                }
                if os.path.exists(output_filename):
                    with open(output_filename, "r") as infile:
                        existing_answers = json.load(infile)
                    existing_answers.append(predicted_answer)
                else:
                    existing_answers = [predicted_answer]
                # Save the updated answers to the JSON file
                with open(output_filename, "w") as outfile:
                    json.dump(existing_answers, outfile, indent=4)
            # break
        # print('predicted_answer \n', predicted_answers )
        # exit()

    




if __name__ == "__main__":
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    train()