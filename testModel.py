import os
import torch
import sys
from torch.utils.data import DataLoader
from torchvision import transforms

from IMDBWIKIDataset import IMDBWIKIDataset
from oldResnetModel import resnet34
from resnetModel import resnet

if __name__ == "__main__":

    # Path variables
    DATASETS_PATH = "../Datasets/"
    PREPROCESSED_IMAGES_PATH = DATASETS_PATH + "Preprocessed/Images/"
    PREPROCESSED_CSV_PATH = DATASETS_PATH + "Preprocessed/CSVs/"
    TRAIN_CSV_PATH = PREPROCESSED_CSV_PATH + "train_dataset.csv"
    VALID_CSV_PATH = PREPROCESSED_CSV_PATH + "valid_dataset.csv"
    TEST_CSV_PATH = PREPROCESSED_CSV_PATH + "test_dataset.csv"

    OUT_PATH = "../TrainedModels/Model1/"
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    # Make results reproducible
    #torch.backends.cudnn.deterministic = True
    RANDOM_SEED = 1

    NUM_WORKERS = 3  # Number of processes in charge of preprocessing batches
    DEVICE = torch.device("cuda:0")
    IMP_WEIGHT = 0

    # Logging
    LOGFILE = os.path.join(OUT_PATH, 'testing.log')
    TEST_PREDICTIONS = os.path.join(OUT_PATH, 'test_predictions.log')
    TEST_ALLPROBAS = os.path.join(OUT_PATH, 'test_allprobas.tensor')

    header = []
    header.append('PyTorch Version: %s' % torch.__version__)
    header.append('CUDA device available: %s' % torch.cuda.is_available())
    header.append('Using CUDA device: %s' % DEVICE)
    header.append('Random Seed: %s' % RANDOM_SEED)
    header.append('Task Importance Weight: %s' % IMP_WEIGHT)
    header.append('Output Path: %s' % OUT_PATH)
    header.append('Script: %s' % sys.argv[0])

    with open(LOGFILE, 'w') as f:
        for entry in header:
            print(entry)
            f.write('%s\n' % entry)
            f.flush()


    # Architecture
    NUM_AGE_CLASSES = 4  # Four classes with ages (13-24),(25-34),(35-49),(50+)
    BATCH_SIZE = 256
    GRAYSCALE = False


    ###################
    # Dataset
    ###################

    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.RandomCrop((120, 120)),
                                           transforms.ToTensor()])

    train_dataset = IMDBWIKIDataset(csv_path=TRAIN_CSV_PATH,
                                    img_dir=PREPROCESSED_IMAGES_PATH,
                                    NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                    transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.CenterCrop((120, 120)),
                                           transforms.ToTensor()])

    test_dataset = IMDBWIKIDataset(csv_path=TEST_CSV_PATH,
                                   img_dir=PREPROCESSED_IMAGES_PATH,
                                   NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                   transform=custom_transform2)

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

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

    ##########################
    # MODEL
    ##########################

    model = resnet34(NUM_AGE_CLASSES, GRAYSCALE)
    model.to(DEVICE)


    def compute_stats(model, data_loader, device):
        age_mae, age_mse, age_acc, gender_acc, overall_acc, num_examples = 0, 0, 0, 0, 0, 0
        age_stats, gender_stats = [], []

        for i in range(NUM_AGE_CLASSES):
            age_stats.append({"total_count": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0})
        for i in range(2):
            gender_stats.append({"total_count": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0})


        for i, (images, age_labels, age_levels, gender_labels) in enumerate(data_loader):

            images = images.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            age_logits, age_probas, gender_probas = model(images)
            predicted_age_levels = age_probas > 0.5
            predicted_age_labels = torch.sum(predicted_age_levels, dim=1)
            predicted_gender_labels = (gender_probas > 0.5).squeeze(1)

            for j in range(NUM_AGE_CLASSES):
                age_stats[j]["total_count"] += torch.sum(age_labels == j)
                age_stats[j]["true_positives"] += torch.sum(torch.logical_and(age_labels == j, predicted_age_labels == j))
                age_stats[j]["false_positives"] += torch.sum(torch.logical_and(age_labels != j, predicted_age_labels == j))
                age_stats[j]["false_negatives"] += torch.sum(torch.logical_and(age_labels == j, predicted_age_labels != j))
            for j in range(2):
                gender_stats[j]["total_count"] += torch.sum(gender_labels == j)
                gender_stats[j]["true_positives"] += torch.sum(torch.logical_and(gender_labels == j, predicted_gender_labels == j))
                gender_stats[j]["false_positives"] += torch.sum(torch.logical_and(gender_labels != j, predicted_gender_labels == j))
                gender_stats[j]["false_negatives"] += torch.sum(torch.logical_and(gender_labels == j, predicted_gender_labels != j))

            correct_age_preds = predicted_age_labels == age_labels
            correct_gender_preds = predicted_gender_labels == gender_labels

            num_examples += age_labels.size(0)
            age_mae += torch.sum(torch.abs(predicted_age_labels - age_labels))
            age_mse += torch.sum((predicted_age_labels - age_labels)**2)
            age_acc += torch.sum(correct_age_preds)
            gender_acc += torch.sum(correct_gender_preds)
            overall_acc += torch.sum(torch.logical_and(correct_gender_preds, correct_age_preds))

        age_mae = age_mae.float() / num_examples
        age_mse = age_mse.float() / num_examples
        age_acc = age_acc.float() / num_examples
        gender_acc = gender_acc.float() / num_examples
        overall_acc = overall_acc.float() / num_examples

        for i in range(NUM_AGE_CLASSES):
            age_stats[i]["precision"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_positives"])
            age_stats[i]["recall"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_negatives"])
            age_stats[i]["f1_score"] = 2 * (age_stats[i]["precision"] * age_stats[i]["recall"]) / (age_stats[i]["precision"] + age_stats[i]["recall"])

        for i in range(2):
            gender_stats[i]["precision"] = gender_stats[i]["true_positives"] / (gender_stats[i]["true_positives"] + gender_stats[i]["false_positives"])
            gender_stats[i]["recall"] = gender_stats[i]["true_positives"] / (gender_stats[i]["true_positives"] + gender_stats[i]["false_negatives"])
            gender_stats[i]["f1_score"] = 2 * (gender_stats[i]["precision"] * gender_stats[i]["recall"]) / (gender_stats[i]["precision"] + gender_stats[i]["recall"])

        print("############AGE STATS############")
        print("AGE MAE:\t", age_mae, "\tAGE MSE:\t", age_mse, "\tAGE ACC:\t", age_acc)
        for i in range(NUM_AGE_CLASSES):
            print("CLASS ", i, ":\tPRECISION:\t", age_stats[i]["precision"], "\tRECALL:\t", age_stats[i]["recall"], "\tF1-SCORE:\t", age_stats[i]["f1_score"], "\tSUPPORT:\t", age_stats[i]["total_count"])

        print("############GENDER STATS############")
        print("FEMALES:\tPRECISION:\t", gender_stats[0]["precision"], "\tRECALL:\t", gender_stats[0]["recall"], "\tF1-SCORE:\t", gender_stats[0]["f1_score"], "\tSUPPORT:\t", gender_stats[0]["total_count"])
        print("MALES:\tPRECISION:\t", gender_stats[1]["precision"], "\tRECALL:\t", gender_stats[1]["recall"], "\tF1-SCORE:\t", gender_stats[1]["f1_score"], "\tSUPPORT:\t", gender_stats[0]["total_count"])
        print("GENDER ACC:\t", gender_acc)

        print("############OVERALL STATS############")
        print("OVERALL ACC:\t", overall_acc)



    model.load_state_dict(torch.load(os.path.join(OUT_PATH, 'best_model.pt')))
    model.eval()
    with torch.set_grad_enabled(False):
        print("##############################")
        print("##########VALIDATION##########")
        print("##############################")
        compute_stats(model, valid_loader, device=DEVICE)

        print("##############################")
        print("###########TESTING############")
        print("##############################")
        compute_stats(model, test_loader, device=DEVICE)

        print("##############################")
        print("###########TRAINING###########")
        print("##############################")
        compute_stats(model, train_loader, device=DEVICE)