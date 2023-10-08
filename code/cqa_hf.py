import torch
from transformers import AutoProcessor, Pix2StructForConditionalGeneration
from dataset.rqa_dataset import RQADataset
from torch.utils.data import DataLoader
from collections import defaultdict
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import argparse
import tqdm
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

parser = argparse.ArgumentParser(description="RQA")
parser.add_argument("-img_dir", "--img_dir", help="Path to the images", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/all_test_train_22/images')
parser.add_argument("-cjson_dir", "--cjson_dir", help="Path to chart jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/all_test_train_22/jsons')
parser.add_argument("-json_dir", "--json_dir", help="Path to the qa jsons", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/combined')
parser.add_argument("-batch_size", "--batch_size", default=20)
# parser.add_argument("-train", "--train", default=True)
parser.add_argument("-train", "--train", default=False)
parser.add_argument("-output", "--output", help="Path to the output file", default='/home/csgrad/sahmed9/reps/RealCQA/code/outputsisi/')
# parser.add_argument("-filter_list", "--filter_list", help="Filter File list for Test ID", default=None)
parser.add_argument("-filter_list", "--filter_list", help="Filter File list for Test ID", default='/home/csgrad/sahmed9/reps/chartinfo-cqa/dataset/cqa_22_with_id/test_filenames.txt')
# parser.add_argument("-unique_qa_ids", "--unique_qa_ids",help="Filter File list for already done ID", default=None)
parser.add_argument("-q_done_list", "--q_done_list",help="Filter File list for already done ID", default="/home/csgrad/sahmed9/reps/RealCQA/output_matcha_plotqav1/unique_qa_ids.txt")

# parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
# parser.add_argument("--master_addr", default="localhost")
# parser.add_argument("--master_port", default="43975")

args = parser.parse_args()

def train():
    # Initialize distributed training
    args.local_rank = int(os.environ['LOCAL_RANK'])
    # args.q_done_list = None
    
    dist.init_process_group(backend='gloo')
    torch.cuda.set_device(int(args.local_rank))

    print('Start Train')
    print('ARGSPACE')
    print(args)

    m_ = ['google/matcha-chartqa', 'google/matcha-plotqa-v1', 'google/matcha-base','google/matcha-plotqa-v2']

    m_ = ['google/matcha-plotqa-v1', 'google/matcha-plotqa-v2']
    # m_ = ['google/matcha-plotqa-v2']

    rqa_dataset = RQADataset(args, transform=None)
    for model_name in m_:
        model = Pix2StructForConditionalGeneration.from_pretrained(model_name).to(args.local_rank)
        model = DistributedDataParallel(model)
        processor = AutoProcessor.from_pretrained(model_name)
        print('Starting MODEL :: ', model_name)
        output_dir = os.path.join(args.output, model_name)
        os.makedirs(output_dir, exist_ok=True)
        predicted_answers = defaultdict(list)
        data_loader = DataLoader(rqa_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=rqa_dataset.custom_collate)
        for data_batch in tqdm.tqdm(data_loader):
            img, q, a, qa_id, pmc_id = data_batch['i'], data_batch['q'], data_batch['a'], data_batch['qa_'], data_batch['p_']
            inputs = processor(images=img, text=q, return_tensors="pt").to(args.local_rank)
            predictions = model.module.generate(**inputs, max_new_tokens=512)
            for i in range(len(pmc_id)):
                output_filename = os.path.join(output_dir, f"{'__'+str(args.local_rank)+'__'+pmc_id[i]}.json")
                predicted_answer = {
                    "qa_id": qa_id[i],
                    "predicted_answer": processor.decode(predictions[i], skip_special_tokens=True),
                }
                if os.path.exists(output_filename):
                    if os.path.getsize(output_filename) > 0:
                        with open(output_filename, "r") as infile:
                            existing_answers = json.load(infile)
                    else:
                        existing_answers = []
                else:
                    existing_answers = []
                existing_answers.append(predicted_answer)
                with open(output_filename, "w") as outfile:
                    json.dump(existing_answers, outfile, indent=4)

if __name__ == "__main__":
    train()






