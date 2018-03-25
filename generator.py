def load_processed_data(dirpath):
    # Reading in the data
    patient_ids = []
    images = []
    for i, filename in enumerate(os.listdir(dirpath)):
        if 'csv' in filename:
            continue
        if i > 160:
            break
        patient_ids.append(filename[8:-4])
        images.append(np.load(dirpath + '/' + filename))
        print('Loading image {}'.format(i))
    return images, patient_ids

def transform_images(images, dim_length):
    resized = np.stack([scipy.ndimage.interpolation.zoom(arr, dim_length / 200)
                    for arr in images])
    print('Resized data')
    normalized = transforms.normalize(resized)
    print('Normalized data')
    return np.expand_dims(normalized, axis=4)


def load_and_transform(dirpath, dim_length):
    images, patient_ids = load_processed_data(dirpath)
    labels = pd.read_excel('/home/lukezhu/data/ELVOS/elvos_meta_drop1.xls')
    print('Loaded data')

    X = transform_images(images, dim_length)
    y = np.zeros(len(patient_ids))
    for _, row in labels.sample(frac=1).iterrows():
        for i, id_ in enumerate(patient_ids):
            if row['PatientID'] == id_:
                y[i] = (row['ELVO status'] == 'Yes')
    print('Parsed labels')
    print('Transformed data')
    return X, y


