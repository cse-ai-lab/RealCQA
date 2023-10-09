import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import time 
import os 
import json 

# Create a custom dataset class
class RQADataset(Dataset):
    def __init__(self, data, transform=None):
        self.img_dir        = data.img_dir
        self.json_dir       = data.json_dir
        self.json_files     = os.listdir(self.json_dir)
        self.q_done_list    = data.q_done_list
        self.filter_test_id = data.filter_list
        self.train          = data.train
        self.transform = transform
        if self.transform is None : 
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # Resize the image to a specific size
                # transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                # transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                # transforms.ToTensor(),  # Convert the image to a PyTorch tensor
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
            ])
        self.questions = []
        if self.filter_test_id is not None :
            # print('Test Data ID from :: ', self.filter_test_id)
            f = open(self.filter_test_id, 'r')
            self.filter_test_id = f.readlines()
            self.filter_test_id = [text.strip() for text in self.filter_test_id]
            
            
        self.create_question()
        print('Sample :', self.questions[0])
        print('Total Questions', len(self.questions))
        if self.q_done_list is not None : 
            self.questions = self.qa_id_done()
            print('Removed done Total Questions', len(self.questions))
            
    
    def qa_id_done(self) :
        done_id = open(self.q_done_list, 'r')
        done_id = done_id.readlines() 
        done_id = [text.strip() for text in done_id]
        return [item for item in self.questions if item['qa_id'] not in done_id]




    def create_question(self):
        start_time = time.time()
        print('\n In create questions')
        print('Total Images', len(self.json_files))
        print('self.train', self.train)
        unused_count = 0
        for js in self.json_files : 
            if self.filter_test_id is not None :
                # print('Test Data ID from :: ', self.filter_test_id)
                # f = open(self.filter_test_id, 'r')
                # self.filter_test_id = f.readlines()
                # self.filter_test_id = [text.strip() for text in self.filter_test_id]
            
                # print(js, self.filter_test_id[:4]) 
                if js[:-5] not in self.filter_test_id:
                    unused_count+=1
                    continue
                    
            #         # print('Filterintng', js)
            #             if not self.train:
            #                 continue
            #             else : 
            #                 print('Exception test in train')
            #         # continue
            #     else:
            #             if not self.train:
            #                 continue
            #             else : 
            #                 print('Exception test in train')
            #                 exit()
            # else : 
            #     print('Warning test ID not provided using all images in directory')

            jsn_list = json.load(open(os.path.join(self.json_dir, js), 'r'))
            self.questions.extend(jsn_list)  
        start_time = time.time() - start_time
        print(f"Elapsed time to create qs: {start_time} seconds ={start_time/60} minutes")
        print('Total unused/used images:', unused_count, '/', len(self.json_files)-unused_count)


    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        return self.load_data(idx)

    def load_data(self, idx):
        # print('In Load Data')
        q_block  = self.questions[idx]
        qa_id    = q_block['qa_id']
        pmc_id   = q_block['PMC_ID']
        answer   = q_block['answer']
        question = q_block['question']
        image    = Image.open(os.path.join(self.img_dir, pmc_id+'.jpg'))
        # print('i, q, a, pmc, qa', image, question, answer, qa_id, pmc_id) 
        if self.transform:
            image = self.transform(image)      
        return {'i' :image, 'q': question, 'a': answer, 'qa_': qa_id, 'p_': pmc_id}
    @staticmethod
    def custom_collate(batch):
        # Separate text and numeric data
        q = [item['q'] for item in batch]
        a = [item['a'] for item in batch]
        p_ = [item['p_'] for item in batch]
        qa_ = [item['qa_'] for item in batch]
        i = [item['i'] for item in batch]
        return {'i' :i, 'q': q, 'a': a, 'qa_':qa_ , 'p_':p_}
