import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

from IMDBWIKIDataset import IMDBWIKIDataset
from AgeOnlyResnetModel import AgeOnlyResnet

def trainAgeOnlyModel(ResNetSize, preprocessedFolderName, outputFolderName):

    # if __name__ == "__main__":

        ##########################
        # SETTINGS
        ##########################

        # Path variables
        DATASETS_PATH = "../Datasets/"
        PREPROCESSED_IMAGES_PATH = DATASETS_PATH + preprocessedFolderName + "/Images/"
        PREPROCESSED_CSV_PATH = DATASETS_PATH + preprocessedFolderName + "/CSVs/"
        TRAIN_CSV_PATH = PREPROCESSED_CSV_PATH + "train_dataset.csv"
        VALID_CSV_PATH = PREPROCESSED_CSV_PATH + "valid_dataset.csv"
        OUT_PATH = "../TrainedModels/" + outputFolderName + "/"
        if not os.path.exists(OUT_PATH):
            os.mkdir(OUT_PATH)

        # Make results reproducible
        torch.backends.cudnn.deterministic = True
        RANDOM_SEED = 1

        # GPU settings
        NUM_WORKERS = 8  # Number of processes in charge of preprocessing batches
        DATA_PARALLEL = True
        CUDA_DEVICE = 0
        if DATA_PARALLEL:
            DEVICE = torch.device("cuda")
        else:
            DEVICE = torch.device("cuda:" + str(CUDA_DEVICE))


        IMP_WEIGHT = 0
        EARLY_STOPPING_PATIENCE = 30

        # Hyperparameters
        learning_rate = 0.0005
        num_epochs = 200
        BATCH_SIZE = 170

        # Architecture
        NUM_AGE_CLASSES = 4  # Four classes with ages (13-24),(25-34),(35-49),(50+)
        RESNET_SIZE = ResNetSize
        # GRAYSCALE = False

        # Define task importance, used to prioritize the loss of certain classes
        imp = torch.ones(NUM_AGE_CLASSES-1, dtype=torch.float)
        imp = imp.to(DEVICE)

        ##########################
        # LOGGING
        ##########################

        LOGFILE = os.path.join(OUT_PATH, 'training.log')

        header = []
        header.append('PyTorch Version: %s' % torch.__version__)
        header.append('CUDA device available: %s' % torch.cuda.is_available())
        header.append('Using CUDA device: %s' % DEVICE)
        header.append('Random Seed: %s' % RANDOM_SEED)
        header.append('Task Importance Weight: %s' % IMP_WEIGHT)
        header.append('Output Path: %s' % OUT_PATH)
        header.append('Script: %s' % sys.argv[0])
        header.append('Data Augmentation: Horizontal Flipping')
        header.append('ResNetSize: %s' % RESNET_SIZE)
        header.append('Batch Size: %s' % BATCH_SIZE)

        with open(LOGFILE, 'w') as f:
            for entry in header:
                print(entry)
                f.write('%s\n' % entry)
                f.flush()


        ###################
        # Dataset
        ###################

        custom_transform = transforms.Compose([transforms.Resize((240, 240)),
                                               transforms.RandomCrop((224, 224)),
                                               transforms.RandomHorizontalFlip(p=0.5),
                                               transforms.ToTensor()])

        train_dataset = IMDBWIKIDataset(csv_path=TRAIN_CSV_PATH,
                                        img_dir=PREPROCESSED_IMAGES_PATH,
                                        NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                        transform=custom_transform)

        custom_transform2 = transforms.Compose([transforms.Resize((240, 240)),
                                               transforms.CenterCrop((224, 224)),
                                               transforms.ToTensor()])

        valid_dataset = IMDBWIKIDataset(csv_path=VALID_CSV_PATH,
                                        img_dir=PREPROCESSED_IMAGES_PATH,
                                        NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                        transform=custom_transform2)

        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=NUM_WORKERS)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=NUM_WORKERS)

        ##########################
        # MODEL
        ##########################

        # model = resnet34(NUM_AGE_CLASSES, GRAYSCALE)
        model = AgeOnlyResnet(RESNET_SIZE, NUM_AGE_CLASSES)
        if DATA_PARALLEL:
            model = nn.DataParallel(model)
        model.to(DEVICE)

        ###########################################
        # COST FUNCTIONS
        ###########################################

        def age_cost_fn(age_logits, age_levels, imp):
            val = (-torch.sum((F.logsigmoid(age_logits)*age_levels
                              + (F.logsigmoid(age_logits) - age_logits)*(1-age_levels))*imp,
                   dim=1))
            return torch.mean(val)

        ###########################################
        # OPTIMIZER
        ###########################################

        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        def compute_stats(model, data_loader, device):
            age_mae, age_mse, age_acc, overall_acc, num_examples, valid_cost = 0, 0, 0, 0, 0, 0
            for i, (images, age_labels, age_levels, gender_labels) in enumerate(data_loader):

                images = images.to(device)
                age_labels = age_labels.to(device)
                age_levels = age_levels.to(device)


                age_logits, age_probas = model(images)
                age_cost = age_cost_fn(age_logits, age_levels, imp)
                valid_cost += age_cost

                predicted_age_levels = age_probas > 0.5
                predicted_age_labels = torch.sum(predicted_age_levels, dim=1)

                correct_age_preds = predicted_age_labels == age_labels

                num_examples += age_labels.size(0)
                age_mae += torch.sum(torch.abs(predicted_age_labels - age_labels))
                age_mse += torch.sum((predicted_age_labels - age_labels)**2)
                age_acc += torch.sum(correct_age_preds)


            age_mae = age_mae.__float__() / num_examples
            age_mse = age_mse.__float__() / num_examples
            age_acc = age_acc.__float__() / num_examples

            return age_mae, age_mse, age_acc, valid_cost


        start_time = time.time()

        best_mae, best_rmse, best_age_acc, best_overall_acc, best_epoch, best_valid_cost = 999, 999, 0, 0, 0, 99999999
        early_stop_counter = 0  # used to count the number of times improvements have stalled on the validation dataset
        for epoch in range(num_epochs):
            model.train()
            for batch_idx, (images, age_labels, age_levels, gender_labels) in enumerate(train_loader):

                images = images.to(DEVICE)
                age_levels = age_levels.to(DEVICE)

                # FORWARD AND BACK PROP
                age_logits, age_probas = model(images)
                age_cost = age_cost_fn(age_logits, age_levels, imp)
                cost = age_cost
                optimizer.zero_grad()
                cost.backward()

                # UPDATE MODEL PARAMETERS
                optimizer.step()

                # LOGGING
                s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' % (epoch + 1, num_epochs, batch_idx, len(train_dataset) // BATCH_SIZE, cost))
                if not batch_idx % 100:
                    print(s)
                    with open(LOGFILE, 'a') as f:
                        f.write('%s\n' % s)

            model.eval()
            with torch.set_grad_enabled(False):
                valid_mae, valid_mse, valid_age_acc, valid_cost = compute_stats(model, valid_loader, device=DEVICE)

            if valid_cost < best_valid_cost:
                best_mae, best_rmse, best_age_acc, best_epoch, best_valid_cost = valid_mae, torch.sqrt(valid_mse), valid_age_acc, epoch, valid_cost
                early_stop_counter = 0
                ########## SAVE MODEL #############
                if DATA_PARALLEL:
                    torch.save(model.module.state_dict(), os.path.join(OUT_PATH, 'best_model.pt'))
                else:
                    torch.save(model.state_dict(), os.path.join(OUT_PATH, 'best_model.pt'))
            else:
                early_stop_counter += 1
                if early_stop_counter > EARLY_STOPPING_PATIENCE:
                    with open(LOGFILE, 'a') as f:
                        s = "IMPROVEMENT ON VALIDATION SET HAS STALLED...INITIATING EARLY-STOPPING"
                        print(s)
                        f.write('%s\n' % s)
                        return best_valid_cost


            s = 'STATS: | Current Valid: MAE=%.4f,MSE=%.4f,AGE_ACC=%.4f, VALID_LOSS=%.4f, EPOCH=%d | ' \
                'Best Valid :MAE=%.4f,MSE=%.4f,AGE_ACC=%.4f, VALID_LOSS=%.4f, EPOCH=%d' % (
                valid_mae, torch.sqrt(valid_mse), valid_age_acc, valid_cost, epoch + 1, best_mae, best_rmse, best_age_acc, best_valid_cost, best_epoch + 1)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

            s = 'Time elapsed: %.4f min' % ((time.time() - start_time)/60)
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

        return best_valid_cost
