    for rate in [0.5, 0.8, 1]:
            for k in [5, 7, 9, 10, 11, 13, 15]:
                t1 = time.time()
                tmp_res = []
                for jpg_name in ok_jpg_name:
                    jpg_path = "pet_dataset/" + jpg_name
                    output_path = "reslut_pet_dataset/exp2/dbscan_and_kmeans/300/take1/" + jpg_path[12:]
                    seg_path = "annotations/annotations/trimaps/" + jpg_name[:-3] + "png"
                    values, p, pred, masks, base_img, masked_img, mapped_array = exp.calc_prob_save_img_by_dbscan_and_kmeans(input_path=jpg_path, output_path=output_path, border=300, eps=15, min_samples=5, k=k, value_border=(1 / k) * rate, save=True)
                    exp2_values = cal_val_from_mappedarray(mapped_array=mapped_array, seg_path=seg_path)
                    exp2_values.append(jpg_name)
                    tmp_res.append(exp2_values)
                t2 = time.time()
                td = t2 - t1
                key = (rate, k)
                exp2_res[key] = tmp_res
                tds[key] = td
                print((key))