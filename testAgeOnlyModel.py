import os
import torch
import xlsxwriter
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from CustomDataset import CustomDataset
from resnetModel import resnet
from AgeOnlyResnetModel import AgeOnlyResnet

def testAgeOnlyModel(ResNetSize, preprocessedFolderName, outputFolderName, MIN_AGE, MAX_AGE, AGE_SEGMENTS_EDGES):


    ##########################
    # SETTINGS
    ##########################

    # Path variables
    DATASETS_PATH = "../Datasets/"
    PREPROCESSED_IMAGES_PATH = DATASETS_PATH + preprocessedFolderName + "/Images/"
    PREPROCESSED_CSV_PATH = DATASETS_PATH + preprocessedFolderName + "/CSVs/"
    TRAIN_CSV_PATH = PREPROCESSED_CSV_PATH + "train_dataset.csv"
    VALID_CSV_PATH = PREPROCESSED_CSV_PATH + "valid_dataset.csv"
    TEST_CSV_PATH = PREPROCESSED_CSV_PATH + "test_dataset.csv"
    OUT_PATH = "../TrainedModels/" + outputFolderName + "/"
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)

    # Make results reproducible
    torch.backends.cudnn.deterministic = True

    NUM_WORKERS = 3  # Number of processes in charge of preprocessing batches
    DATA_PARALLEL = True
    CUDA_DEVICE = 0
    if DATA_PARALLEL:
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cuda:" + str(CUDA_DEVICE))

    # Hyperparameters
    BATCH_SIZE = 256

    # Architecture
    NUM_AGE_CLASSES = MAX_AGE - MIN_AGE
    RESNET_SIZE = ResNetSize


    ###################
    # Dataset
    ###################

    custom_transform = transforms.Compose([transforms.Resize((240, 240)),
                                           transforms.RandomCrop((224, 224)),
                                           transforms.ToTensor()])

    train_dataset = CustomDataset(csv_path=TRAIN_CSV_PATH,
                                  img_dir=PREPROCESSED_IMAGES_PATH,
                                  NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                  MIN_AGE=MIN_AGE,
                                  transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.ToTensor()])

    test_dataset = CustomDataset(csv_path=TEST_CSV_PATH,
                                 img_dir=PREPROCESSED_IMAGES_PATH,
                                 NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                 MIN_AGE=MIN_AGE,
                                 transform=custom_transform2)

    valid_dataset = CustomDataset(csv_path=VALID_CSV_PATH,
                                  img_dir=PREPROCESSED_IMAGES_PATH,
                                  NUM_AGE_CLASSES=NUM_AGE_CLASSES,
                                  MIN_AGE=MIN_AGE,
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
    #COMPUTE STATS FUNCTION
    ##########################
    def compute_stats(model, data_loader, dataset_name, device):
        age_mae, age_mse, age_acc, overall_acc, num_examples = 0, 0, 0, 0, 0
        age_stats = []

        for i in range(NUM_AGE_CLASSES):
            age_stats.append({"total_count": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0})
        # for i in range(2):
        #     gender_stats.append({"total_count": 0, "true_positives": 0, "false_positives": 0, "false_negatives": 0})


        for i, (images, age_labels, age_levels, gender_labels) in enumerate(data_loader):

            # Move batch to device
            images = images.to(device)
            age_labels = age_labels.to(device)
            gender_labels = gender_labels.to(device)

            # Get model predictions
            age_logits, age_probas = model(images)
            predicted_age_levels = age_probas > 0.5
            predicted_age_labels = torch.sum(predicted_age_levels, dim=1) + MIN_AGE
            # predicted_gender_labels = (gender_probas > 0.5).squeeze(1)

            # Get Age Segments from Age Labels
            predicted_age_segments = torch.zeros_like(predicted_age_labels)
            age_segments = torch.zeros_like(predicted_age_labels)
            for id, segment_edge in enumerate(AGE_SEGMENTS_EDGES):
                predicted_age_segments[predicted_age_labels >= segment_edge] = id + 1
                age_segments[age_labels >= segment_edge] = id + 1

            for j in range(NUM_AGE_CLASSES):
                age_stats[j]["total_count"] += torch.sum(age_labels == j)
                age_stats[j]["true_positives"] += torch.sum(torch.logical_and(age_segments == j, predicted_age_segments == j))
                age_stats[j]["false_positives"] += torch.sum(torch.logical_and(age_segments != j, predicted_age_segments == j))
                age_stats[j]["false_negatives"] += torch.sum(torch.logical_and(age_segments == j, predicted_age_segments != j))
            # for j in range(2):
            #     gender_stats[j]["total_count"] += torch.sum(gender_labels == j)
            #     gender_stats[j]["true_positives"] += torch.sum(torch.logical_and(gender_labels == j, predicted_gender_labels == j))
            #     gender_stats[j]["false_positives"] += torch.sum(torch.logical_and(gender_labels != j, predicted_gender_labels == j))
            #     gender_stats[j]["false_negatives"] += torch.sum(torch.logical_and(gender_labels == j, predicted_gender_labels != j))

            correct_age_preds = predicted_age_segments == age_segments
            # correct_gender_preds = predicted_gender_labels == gender_labels

            num_examples += age_labels.size(0)
            age_mae += torch.sum(torch.abs(predicted_age_labels - age_labels))
            age_mse += torch.sum((predicted_age_labels - age_labels)**2)
            age_acc += torch.sum(correct_age_preds)
            # gender_acc += torch.sum(correct_gender_preds)
            # overall_acc += torch.sum(torch.logical_and(correct_gender_preds, correct_age_preds))

        age_mae = age_mae.float() / num_examples
        age_mse = age_mse.float() / num_examples
        age_acc = age_acc.float() / num_examples
        # gender_acc = gender_acc.float() / num_examples
        # overall_acc = overall_acc.float() / num_examples

        for i in range(NUM_AGE_CLASSES):
            age_stats[i]["precision"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_positives"])
            age_stats[i]["recall"] = age_stats[i]["true_positives"] / (age_stats[i]["true_positives"] + age_stats[i]["false_negatives"])
            age_stats[i]["f1_score"] = 2 * (age_stats[i]["precision"] * age_stats[i]["recall"]) / (age_stats[i]["precision"] + age_stats[i]["recall"])

        # for i in range(2):
        #     gender_stats[i]["precision"] = gender_stats[i]["true_positives"] / (gender_stats[i]["true_positives"] + gender_stats[i]["false_positives"])
        #     gender_stats[i]["recall"] = gender_stats[i]["true_positives"] / (gender_stats[i]["true_positives"] + gender_stats[i]["false_negatives"])
        #     gender_stats[i]["f1_score"] = 2 * (gender_stats[i]["precision"] * gender_stats[i]["recall"]) / (gender_stats[i]["precision"] + gender_stats[i]["recall"])


        # WRITE RESULTS TO EXCEL
        workbook = xlsxwriter.Workbook(OUT_PATH + dataset_name + 'Results.xlsx')
        worksheet = workbook.add_worksheet()
        row = 0
        col = 0

        # Prepare column headings
        worksheet.write(row, col + 1, "Precision")
        worksheet.write(row, col + 2, "Recall")
        worksheet.write(row, col + 3, "F1-Score")
        worksheet.write(row, col + 4, "Support")
        row += 1

        # Write Age Results
        for i in range(NUM_AGE_CLASSES):
            worksheet.write(row, col, "CLASS" + str(i) + ":")
            worksheet.write(row, col + 1, age_stats[i]["precision"].__float__())
            worksheet.write(row, col + 2, age_stats[i]["recall"].__float__())
            worksheet.write(row, col + 3, age_stats[i]["f1_score"].__float__())
            worksheet.write(row, col + 4, age_stats[i]["total_count"].__float__())
            row += 1
        worksheet.write(row, col, "Age Acc:")
        worksheet.write(row, col + 1, age_acc)
        worksheet.write(row, col + 2, "Age MAE:")
        worksheet.write(row, col + 3, age_mae)
        worksheet.write(row, col + 4, "Age MSE:")
        worksheet.write(row, col + 5, age_mse)
        row += 2

        # # Write Gender Results
        # for i in range(2):
        #     if i == 0:
        #         gender = "FEMALES:"
        #     else:
        #         gender = "MALES:"
        #     worksheet.write(row, col, gender)
        #     worksheet.write(row, col + 1, gender_stats[i]["precision"].__float__())
        #     worksheet.write(row, col + 2, gender_stats[i]["recall"].__float__())
        #     worksheet.write(row, col + 3, gender_stats[i]["f1_score"].__float__())
        #     worksheet.write(row, col + 4, gender_stats[i]["total_count"].__float__())
        #     row += 1
        # worksheet.write(row, col, "Gender Acc:")
        # worksheet.write(row, col + 1, gender_acc)
        # row += 2
        #
        # # Write Overall Results
        # worksheet.write(row, col, "Overall Acc:")
        # worksheet.write(row, col + 1, overall_acc)

        # Close xlsx
        workbook.close()

    ##########################
    # MODEL
    ##########################

    model = AgeOnlyResnet(RESNET_SIZE, NUM_AGE_CLASSES)
    model.load_state_dict(torch.load(os.path.join(OUT_PATH, 'best_model.pt')))
    if DATA_PARALLEL:
        model = nn.DataParallel(model)
    model.to(DEVICE)
    model.eval()
    with torch.set_grad_enabled(False):
        print("Validation Dataset...")
        compute_stats(model, valid_loader, "Validation", device=DEVICE)
        print("Testing Dataset...")
        compute_stats(model, test_loader, "Testing", device=DEVICE)
        print("Training Dataset...")
        compute_stats(model, train_loader, "Training", device=DEVICE)
        print("Finished!")