import os
import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from PIL import Image


def get_validation_recalls(r_list, q_list, k_values, gt, eval_dataset=None, print_results=True, faiss_gpu=False,
                           save_retrieval=False,
                           dataset_name='dataset without name ?', index_method='IndexFlatL2'):
    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        if index_method == 'IndexFlatL2':
            faiss_index = faiss.IndexFlatL2(embed_size)
        elif index_method == 'IndexIVFPQ':
            nlist = 100
            m = 8
            quantizer = faiss.IndexFlatL2(embed_size)
            faiss_index = faiss.IndexIVFPQ(quantizer, embed_size, nlist, m, 8)
            faiss_index.train(r_list)
        elif index_method == 'IndexIVFFlat':
            nlist = 100
            quantizer = faiss.IndexFlatL2(embed_size)
            faiss_index = faiss.IndexIVFFlat(quantizer, embed_size, nlist)
            faiss_index.train(r_list)

    # add references
    faiss_index.add(r_list)

    # get the memory consumption for the faiss index
    def get_index_memory(index):
        faiss.write_index(index, './tmp/temp.index')
        file_size = os.path.getsize('./tmp/temp.index')
        os.remove('./tmp/temp.index')
        return file_size / (1024 * 1024)

    index_memory = get_index_memory(faiss_index)
    print(f"\nIndex cost : {index_memory:.2f} MB")

    # search for queries in the index
    start_get_recall_time = time.time()
    _, predictions = faiss_index.search(q_list, max(k_values))
    predictions_new = predictions

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    correct_query_flag_list = []
    correct_query_flag = False
    for q_idx, pred_ in enumerate(predictions_new):

        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred_[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                if i == 0:
                    correct_query_flag = True
                break
        if correct_query_flag:
            correct_query_flag_list.append(True)
        else:
            correct_query_flag_list.append(False)
        correct_query_flag = False

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ['K'] + [str(k) for k in k_values]
        table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name}"))

    # calculate the estimation error
    totol_loss = 0
    for q_idx, pred_ in enumerate(predictions_new):
        db_name = eval_dataset.db_paths[pred_[0]]
        qr_name = eval_dataset.qr_paths[q_idx]
        db_height = float(db_name.split('_')[-2])
        if eval_dataset.dataset_info['dataset_name'] == 'MHFlightDataset':
            qr_height = (float(qr_name.split("@")[4]) - 35) * 15.36 / 12
        elif eval_dataset.dataset_info['dataset_name'] == 'GEStuidioDataset':
            qr_height = (float(qr_name.split("@")[4])) * 0.357 + 5.33
        pred_error = abs(db_height - qr_height)
        totol_loss += pred_error
    print("average totol loss: ", totol_loss / (predictions.shape[0]))

    if save_retrieval:
        # save the viz result
        for q_idx, pred_ in enumerate(predictions_new[::10]):
            q_idx = q_idx * 10
            retrieved_tile_name_0 = eval_dataset.db_paths[pred_[0]]
            retrieved_tile_name_1 = eval_dataset.db_paths[pred_[1]]
            retrieved_tile_name_2 = eval_dataset.db_paths[pred_[2]]
            retrieved_tile_name_3 = eval_dataset.db_paths[pred_[3]]
            retrieved_tile_name_4 = eval_dataset.db_paths[pred_[4]]
            query_image_name = eval_dataset.qr_paths[q_idx]
            retrieved_tile_0 = mpimg.imread(retrieved_tile_name_0)
            retrieved_tile_1 = mpimg.imread(retrieved_tile_name_1)
            retrieved_tile_2 = mpimg.imread(retrieved_tile_name_2)
            retrieved_tile_3 = mpimg.imread(retrieved_tile_name_3)
            retrieved_tile_4 = mpimg.imread(retrieved_tile_name_4)
            pil_image = Image.open(query_image_name)
            query_image = np.array(pil_image)
            plt.subplot(1, 6, 1)
            plt.axis('off')
            plt.imshow(query_image)
            if correct_query_flag_list[q_idx]:
                plt.title("Success", fontsize=14)
            else:
                plt.title("Unsuccess", fontsize=14)
            plt.subplot(1, 6, 2)
            plt.axis('off')
            plt.imshow(retrieved_tile_0)
            plt.title(retrieved_tile_name_0.split('_')[-2])
            plt.subplot(1, 6, 3)
            plt.axis('off')
            plt.imshow(retrieved_tile_1)
            plt.title(retrieved_tile_name_1.split('_')[-2])
            plt.subplot(1, 6, 4)
            plt.axis('off')
            plt.imshow(retrieved_tile_2)
            plt.title(retrieved_tile_name_2.split('_')[-2])
            plt.subplot(1, 6, 5)
            plt.axis('off')
            plt.imshow(retrieved_tile_3)
            plt.title(retrieved_tile_name_3.split('_')[-2])
            plt.subplot(1, 6, 6)
            plt.axis('off')
            plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.2)
            plt.imshow(retrieved_tile_4)
            plt.title(retrieved_tile_name_4.split('_')[-2])
            sub_dir_save_viz_result = './utils/result_viz/{}'.format(dataset_name)
            os.makedirs(sub_dir_save_viz_result, exist_ok=True)
            plt.tight_layout()
            plt.savefig('./utils/result_viz/{}/retrieval_{}.png'.format(dataset_name, str(q_idx).zfill(5)), dpi=300,
                        bbox_inches='tight')

    print('Processing: ', str(time.time() - start_get_recall_time), " seconds")
    return d
