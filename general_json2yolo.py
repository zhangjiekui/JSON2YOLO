import json
import os
import cv2
import pandas as pd
from PIL import Image

from utils import *


# Convert INFOLKS JSON file into YOLO-format labels ----------------------------
def convert_infolks_json(name, files, img_path):
    # Create folders
    path = make_dirs()

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Write images and shapes
    name = path + os.sep + name
    file_id, file_name, wh, cat = [], [], [], []
    for x in tqdm(data, desc='Files and Shapes'):
        f = glob.glob(img_path + Path(x['json_file']).stem + '.*')[0]
        file_name.append(f)
        wh.append(exif_size(Image.open(f)))  # (width, height)
        cat.extend(a['classTitle'].lower() for a in x['output']['objects'])  # categories

        # filename
        with open(name + '.txt', 'a') as file:
            file.write('%s\n' % f)

    # Write *.names file
    names = sorted(np.unique(cat))
    # names.pop(names.index('Missing product'))  # remove
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    for i, x in enumerate(tqdm(data, desc='Annotations')):
        label_name = Path(file_name[i]).stem + '.txt'

        with open(path + '/labels/' + label_name, 'a') as file:
            for a in x['output']['objects']:
                # if a['classTitle'] == 'Missing product':
                #    continue  # skip

                category_id = names.index(a['classTitle'].lower())

                # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                box = np.array(a['points']['exterior'], dtype=np.float32).ravel()
                box[[0, 2]] /= wh[i][0]  # normalize x by width
                box[[1, 3]] /= wh[i][1]  # normalize y by height
                box = [box[[0, 2]].mean(), box[[1, 3]].mean(), box[2] - box[0], box[3] - box[1]]  # xywh
                if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))

    # Split data into train, test, and validate files
    split_files(name, file_name)
    write_data_data(name + '.data', nc=len(names))
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert vott JSON file into YOLO-format labels -------------------------------
def convert_vott_json(name, files, img_path):
    # Create folders
    path = make_dirs()
    name = path + os.sep + name

    # Import json
    data = []
    for file in glob.glob(files):
        with open(file) as f:
            jdata = json.load(f)
            jdata['json_file'] = file
            data.append(jdata)

    # Get all categories
    file_name, wh, cat = [], [], []
    for i, x in enumerate(tqdm(data, desc='Files and Shapes')):
        try:
            cat.extend(a['tags'][0] for a in x['regions'])  # categories
        except:
            pass

    # Write *.names file
    names = sorted(pd.unique(cat))
    with open(name + '.names', 'a') as file:
        [file.write('%s\n' % a) for a in names]

    # Write labels file
    n1, n2 = 0, 0
    missing_images = []
    for i, x in enumerate(tqdm(data, desc='Annotations')):

        f = glob.glob(img_path + x['asset']['name'] + '.jpg')
        if len(f):
            f = f[0]
            file_name.append(f)
            wh = exif_size(Image.open(f))  # (width, height)

            n1 += 1
            if (len(f) > 0) and (wh[0] > 0) and (wh[1] > 0):
                n2 += 1

                # append filename to list
                with open(name + '.txt', 'a') as file:
                    file.write('%s\n' % f)

                # write labelsfile
                label_name = Path(f).stem + '.txt'
                with open(path + '/labels/' + label_name, 'a') as file:
                    for a in x['regions']:
                        category_id = names.index(a['tags'][0])

                        # The INFOLKS bounding box format is [x-min, y-min, x-max, y-max]
                        box = a['boundingBox']
                        box = np.array([box['left'], box['top'], box['width'], box['height']]).ravel()
                        box[[0, 2]] /= wh[0]  # normalize x by width
                        box[[1, 3]] /= wh[1]  # normalize y by height
                        box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2], box[3]]  # xywh

                        if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                            file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
        else:
            missing_images.append(x['asset']['name'])

    print('Attempted %g json imports, found %g images, imported %g annotations successfully' % (i, n1, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Split data into train, test, and validate files
    split_files(name, file_name)
    print('Done. Output saved to %s' % (os.getcwd() + os.sep + path))


# Convert ath JSON file into YOLO-format labels --------------------------------
def convert_ath_json(json_dir):  # dir contains json annotations and images
    # Create folders
    dir = make_dirs()  # output directory

    jsons = []
    for dirpath, dirnames, filenames in os.walk(json_dir):
        for filename in [f for f in filenames if f.lower().endswith('.json')]:
            jsons.append(os.path.join(dirpath, filename))

    # Import json
    n1, n2, n3 = 0, 0, 0
    missing_images, file_name = [], []
    for json_file in sorted(jsons):
        with open(json_file) as f:
            data = json.load(f)

        # # Get classes
        # try:
        #     classes = list(data['_via_attributes']['region']['class']['options'].values())  # classes
        # except:
        #     classes = list(data['_via_attributes']['region']['Class']['options'].values())  # classes

        # # Write *.names file
        # names = pd.unique(classes)  # preserves sort order
        # with open(dir + 'data.names', 'w') as f:
        #     [f.write('%s\n' % a) for a in names]

        # Write labels file
        for i, x in enumerate(tqdm(data['_via_img_metadata'].values(), desc='Processing %s' % json_file)):

            image_file = str(Path(json_file).parent / x['filename'])
            f = glob.glob(image_file)  # image file
            if len(f):
                f = f[0]
                file_name.append(f)
                wh = exif_size(Image.open(f))  # (width, height)

                n1 += 1  # all images
                if len(f) > 0 and wh[0] > 0 and wh[1] > 0:
                    label_file = dir + 'labels/' + Path(f).stem + '.txt'

                    nlabels = 0
                    try:
                        with open(label_file, 'a') as file:  # write labelsfile
                            for a in x['regions']:
                                # try:
                                #     category_id = int(a['region_attributes']['class'])
                                # except:
                                #     category_id = int(a['region_attributes']['Class'])
                                category_id = 0  # single-class

                                # bounding box format is [x-min, y-min, x-max, y-max]
                                box = a['shape_attributes']
                                box = np.array([box['x'], box['y'], box['width'], box['height']],
                                               dtype=np.float32).ravel()
                                box[[0, 2]] /= wh[0]  # normalize x by width
                                box[[1, 3]] /= wh[1]  # normalize y by height
                                box = [box[0] + box[2] / 2, box[1] + box[3] / 2, box[2],
                                       box[3]]  # xywh (left-top to center x-y)

                                if box[2] > 0. and box[3] > 0.:  # if w > 0 and h > 0
                                    file.write('%g %.6f %.6f %.6f %.6f\n' % (category_id, *box))
                                    n3 += 1
                                    nlabels += 1

                        if nlabels == 0:  # remove non-labelled images from dataset
                            os.system('rm %s' % label_file)
                            # print('no labels for %s' % f)
                            continue  # next file

                        # write image
                        img_size = 4096  # resize to maximum
                        img = cv2.imread(f)  # BGR
                        assert img is not None, 'Image Not Found ' + f
                        r = img_size / max(img.shape)  # size ratio
                        if r < 1:  # downsize if necessary
                            h, w, _ = img.shape
                            img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)

                        ifile = dir + 'images/' + Path(f).name
                        if cv2.imwrite(ifile, img):  # if success append image to list
                            with open(dir + 'data.txt', 'a') as file:
                                file.write('%s\n' % ifile)
                            n2 += 1  # correct images

                    except:
                        os.system('rm %s' % label_file)
                        print('problem with %s' % f)

            else:
                missing_images.append(image_file)

    nm = len(missing_images)  # number missing
    print('\nFound %g JSONs with %g labels over %g images. Found %g images, labelled %g images successfully' %
          (len(jsons), n3, n1, n1 - nm, n2))
    if len(missing_images):
        print('WARNING, missing images:', missing_images)

    # Write *.names file
    names = ['knife']  # preserves sort order
    with open(dir + 'data.names', 'w') as f:
        [f.write('%s\n' % a) for a in names]

    # Split data into train, test, and validate files
    split_rows_simple(dir + 'data.txt')
    write_data_data(dir + 'data.data', nc=1)
    print('Done. Output saved to %s' % Path(dir).absolute())


def convert_coco_json(json_dir='../coco/annotations/', use_segments=False,just_compare=False, cls91to80=False, compare_to_orininal_provided_coco_txt=True, decimals = 3, coco_orininal_provided_txt_dir=r'C:\winyolox\coco128_txt\labels'):
    coco80 = coco91_to_coco80_class()
    save_dir = json_dir + r'/txts'
    if not just_compare: # 只是比较是否一致（说明文件已经生成过了）
        save_dir = make_dirs(save_dir)  # output directory

    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob('*.json'),reverse=True):
        fn = Path(save_dir) / 'labels' / json_file.stem.replace('instances_', '')  # folder name
        if not just_compare:
            fn.mkdir()

        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        # images = {'%g' % x['id']: x for x in data['images']}
        images = {str(int(x['id'])): x for x in data['images']}
        print("")
        print(f"在{json_file}中共有{len(images)}张图片！{len(data['annotations'])}个标注。")

        if compare_to_orininal_provided_coco_txt:
            txt_dir = Path(coco_orininal_provided_txt_dir)/json_file.stem.replace('instances_', '')
            coco_txt_files = sorted(txt_dir.resolve().glob('*.txt'))
            if len(coco_txt_files)==0:
                continue
            coco_img_files_names = [x.stem + '.jpg' for x in coco_txt_files]

        # Write labels file
        # """
        if not just_compare:
            for x in tqdm(data['annotations'], desc=f'Annotations {json_file}'):
                if x['iscrowd']:
                    continue
                idx = '%g' % x['image_id']
                img = images[idx]
                h, w, f = img['height'], img['width'], img['file_name']

                if  compare_to_orininal_provided_coco_txt and f in coco_img_files_names:
                    # The COCO box format is [top left x, top left y, width, height]
                    box = np.array(x['bbox'], dtype=np.float64)
                    box[:2] += box[2:] / 2  # xy top-left corner to center
                    box[[0, 2]] /= w  # normalize x
                    box[[1, 3]] /= h  # normalize y

                    # Segments
                    if use_segments:
                        segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                        s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

                    # Write
                    if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                        cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
                        line = cls, *(s if use_segments else box)  # cls, box or segments
                        with open((fn / f).with_suffix('.txt'), 'a') as file:
                            file.write(('%g ' * len(line)).rstrip() % line + '\n')

            if not compare_to_orininal_provided_coco_txt:
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(x['bbox'], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y

                # Segments
                if use_segments:
                    segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
                    s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

                # Write
                if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
                    cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
                    line = cls, *(s if use_segments else box)  # cls, box or segments
                    with open((fn / f).with_suffix('.txt'), 'a') as file:
                        file.write(('%g ' * len(line)).rstrip() % line + '\n')
        # """
        if compare_to_orininal_provided_coco_txt:
            coco_txt_files_writed = sorted(fn.resolve().glob('*.txt'))
            coco_img_files_names_writed = [x.stem + '.jpg' for x in coco_txt_files_writed]


            original_txt_file_dir = coco_txt_files[0].parent
            right=0
            error=0
            warn =0
            for writed_file in coco_txt_files_writed:
                original_txt_file = original_txt_file_dir/ writed_file.name
                s=np.loadtxt(writed_file)
                d=np.loadtxt(original_txt_file)
                # decimals = 3
                try:
                    # c = np.around(s,decimals)==np.around(d,decimals)
                    # comp_result = c.all()
                    comp_result =np.allclose(s,d, atol=1e-04)

                except:
                    warn += 1
                    print(f"拟修正{writed_file=}")
                    # print(f"{c=}")
                    print(f"{s.shape=}")
                    print(f"{d.shape=}")
                    print("修正.......")
                    # C:\winyolox\COCO2017\COCO\annotations\txts\labels\train2017\000000099844.txt
                    #                                                             000000201706.txt
                    _, idx = np.unique(s, axis=0, return_index=True)
                    s_unique = s[np.sort(idx)]

                    print(f"    {s_unique.shape=}")
                    np.savetxt(writed_file, s_unique,fmt="%g", delimiter=" ")
                    s_unique_loaded = np.loadtxt(writed_file)
                    print(f"{    s_unique_loaded.shape=}")
                    comp_result = np.allclose(s_unique_loaded, d, atol=1e-04)

                # comp_result = cmp_file(writed_file, original_txt_file)
                if comp_result:
                    right+=1
                else:
                    error+=1
                    print("不一致的文件：",writed_file)
            print(
                f"目标文件有{len(coco_img_files_names)}个,写入的文件有{len(coco_img_files_names_writed)}个,其中不在的文件名有：{set(coco_img_files_names) - set(coco_img_files_names_writed)}")
            print(f"写入的{len(coco_img_files_names_writed)}个文件中，数值都一致的文件有{right}个,不一致的文件有{error}个！警告[有重复行]的文件有{warn}个，！")
            o_cmp=cmp_file(original_txt_file,original_txt_file)
            w_cmp = cmp_file(writed_file,writed_file)

            print(f"直接比较两个文件的结果{o_cmp=},{w_cmp=}")

# 这个函数暂时保留
def cmp_file(f1, f2):
    st1 = os.stat(f1)
    st2 = os.stat(f2)

    # 比较文件大小
    if st1.st_size != st2.st_size:
        return False

    bufsize = 8*1024
    with open(f1, 'rb') as fp1, open(f2, 'rb') as fp2:
        while True:
            b1 = fp1.read(bufsize)  # 读取指定大小的数据进行比较
            b2 = fp2.read(bufsize)
            if b1 != b2:
                return False
            if not b1:
                return True

if __name__ == '__main__':
    source = 'COCO'
    if source == 'COCO':

        path=         r'C:\winyolox\COCO2017\COCO\annotations'
        coco_txt_dir =r'C:\winyolox\COCO2017\COCO\labels'
        # coco_txt_dir = r'C:\winyolox\coco128_txt\labels' # todo 暂时没有验证集val的label

        convert_coco_json(path, cls91to80=True, just_compare=True,coco_orininal_provided_txt_dir=coco_txt_dir)
        # convert_coco_json('../../Downloads/Objects365')  # directory with *.json

    elif source == 'infolks':  # Infolks https://infolks.info/
        convert_infolks_json(name='out',
                             files='../data/sm4/json/*.json',
                             img_path='../data/sm4/images/')

    elif source == 'vott':  # VoTT https://github.com/microsoft/VoTT
        convert_vott_json(name='data',
                          files='../../Downloads/athena_day/20190715/*.json',
                          img_path='../../Downloads/athena_day/20190715/')  # images folder

    elif source == 'ath':  # ath format
        convert_ath_json(json_dir='../../Downloads/athena/')  # images folder

    # zip results
    # os.system('zip -r ../coco.zip ../coco')
