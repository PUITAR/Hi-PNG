import os
import random
import tarfile
from urllib.request import urlopen, urlretrieve

import h5py
import numpy
from typing import Any, Callable, Dict, Tuple

def download(source_url: str, destination_path: str) -> None:
    """
    Downloads a file from the provided source URL to the specified destination path
    only if the file doesn't already exist at the destination.
    
    Args:
        source_url (str): The URL of the file to download.
        destination_path (str): The local path where the file should be saved.
    """
    if not os.path.exists(destination_path):
        print(f"downloading {source_url} -> {destination_path}...")
        urlretrieve(source_url, destination_path)


def get_dataset_fn(dataset_name: str) -> str:
    """
    Returns the full file path for a given dataset name in the data directory.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        str: The full file path of the dataset.
    """
    if not os.path.exists("data"):
        os.mkdir("data")
    return os.path.join(f"data", f"{dataset_name}.hdf5")


def get_dataset(dataset_name: str) -> Tuple[h5py.File, int]:
    """
    Fetches a dataset by downloading it from a known URL or creating it locally
    if it's not already present. The dataset file is then opened for reading, 
    and the file handle and the dimension of the dataset are returned.
    
    Args:
        dataset_name (str): The name of the dataset.
    
    Returns:
        Tuple[h5py.File, int]: A tuple containing the opened HDF5 file object and
            the dimension of the dataset.
    """
    hdf5_filename = get_dataset_fn(dataset_name)
    try:
        dataset_url = f"https://ann-benchmarks.com/{dataset_name}.hdf5"
        download(dataset_url, hdf5_filename)
    except:
        print(f"Cannot download {dataset_url}")
        if dataset_name in DATASETS:
            print("Creating dataset locally")
            DATASETS[dataset_name](hdf5_filename)

    hdf5_file = h5py.File(hdf5_filename, "r")

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_file.attrs["dimension"]) if "dimension" in hdf5_file.attrs else len(hdf5_file["train"][0])
    return hdf5_file, dimension


def write_output(train: numpy.ndarray, test: numpy.ndarray, fn: str, distance: str, point_type: str = "float", count: int = 100) -> None:
    """
    Writes the provided training and testing data to an HDF5 file. It also computes 
    and stores the nearest neighbors and their distances for the test set using a 
    brute-force approach.
    
    Args:
        train (numpy.ndarray): The training data.
        test (numpy.ndarray): The testing data.
        filename (str): The name of the HDF5 file to which data should be written.
        distance_metric (str): The distance metric to use for computing nearest neighbors.
        point_type (str, optional): The type of the data points. Defaults to "float".
        neighbors_count (int, optional): The number of nearest neighbors to compute for 
            each point in the test set. Defaults to 100.
    """
    # from bruteforce import BruteForceBLAS

    with h5py.File(fn, "w") as f:
        f.attrs["type"] = "dense"
        f.attrs["distance"] = distance
        f.attrs["dimension"] = len(train[0])
        f.attrs["point_type"] = point_type
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)

        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")

        # Create datasets for neighbors and distances
        # neighbors_ds = f.create_dataset("neighbors", (len(test), count), dtype=int)
        # distances_ds = f.create_dataset("distances", (len(test), count), dtype=float)

        # Fit the brute-force k-NN model
        # bf = BruteForceBLAS(distance, precision=train.dtype)
        # bf.fit(train)

        # for i, x in enumerate(test):
        #     if i % 1000 == 0:
        #         print(f"{i}/{len(test)}...")

            # Query the model and sort results by distance
            # res = list(bf.query_with_distances(x, count))
            # res.sort(key=lambda t: t[-1])

            # Save neighbors indices and distances
            # neighbors_ds[i] = [idx for idx, _ in res]
            # distances_ds[i] = [dist for _, dist in res]
    
        # Save data into vecs-format file
        from data_collect.binary import ivecs_save, fvecs_save

        fp = fn.split(".")[0]
        fvecs_save(fp + ".train.fvecs", train)
        fvecs_save(fp + ".test.fvecs", test)
        # ivecs_save(fp + ".gt.ivecs", neighbors_ds[:])
        # print(f"Groundtruth shape: {neighbors_ds[:].shape}")


def train_test_split(X: numpy.ndarray, test_size: int = 10000, dimension: int = None) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Splits the provided dataset into a training set and a testing set.
    
    Args:
        X (numpy.ndarray): The dataset to split.
        test_size (int, optional): The number of samples to include in the test set. 
            Defaults to 10000.
        dimension (int, optional): The dimensionality of the data. If not provided, 
            it will be inferred from the second dimension of X. Defaults to None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: A tuple containing the training set and the testing set.
    """
    from sklearn.model_selection import train_test_split as sklearn_train_test_split

    dimension = dimension if not None else X.shape[1]
    print(f"Splitting {X.shape[0]}*{dimension} into train/test")
    return sklearn_train_test_split(X, test_size=test_size, random_state=1)


def glove(out_fn: str, d: int) -> None:
    import zipfile

    url = "http://nlp.stanford.edu/data/glove.twitter.27B.zip"
    fn = os.path.join("data", "glove.twitter.27B.zip")
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print("preparing %s" % out_fn)
        z_fn = "glove.twitter.27B.%dd.txt" % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(numpy.array(v))
        X = numpy.array(X)
        X_train, X_test = train_test_split(X)
        write_output(X_train, X_test, out_fn, "angular")


def _load_texmex_vectors(f: Any, n: int, k: int) -> numpy.ndarray:
    import struct

    v = numpy.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack("f" * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t: tarfile.TarFile, fn: str) -> numpy.ndarray:
    import struct

    m = t.getmember(fn)
    f = t.extractfile(m)
    (k,) = struct.unpack("i", f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
    fn = os.path.join("data", 'sift.tar.gz')
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "sift/sift_base.fvecs")
        test = _get_irisa_matrix(t, "sift/sift_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def gist(out_fn: str) -> None:
    import tarfile

    url = "ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz"
    fn = os.path.join("data", "gist.tar.tz")
    download(url, fn)
    with tarfile.open(fn, "r:gz") as t:
        train = _get_irisa_matrix(t, "gist/gist_base.fvecs")
        test = _get_irisa_matrix(t, "gist/gist_query.fvecs")
        write_output(train, test, out_fn, "euclidean")


def _load_mnist_vectors(fn: str) -> numpy.ndarray:
    import gzip
    import struct

    print("parsing vectors in %s..." % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d"),
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0] for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = numpy.prod(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0] for j in range(entry_size)])
    return numpy.array(vectors)


def mnist(out_fn: str) -> None:
    fn_train = "data/mnist-train.gz"
    fn_test = "data/mnist-test.gz"
    download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", fn_train)  # noqa
    download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", fn_test)  # noqa
    train = _load_mnist_vectors(fn_train)
    test = _load_mnist_vectors(fn_test)
    write_output(train, test, out_fn, "euclidean")


def fashion_mnist(out_fn: str) -> None:
    fn_train = "data/fashion-mnist-train.gz"
    fn_test = "data/fashion-mnist-test.gz"
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",  # noqa
        fn_train,
    )
    download(
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",  # noqa
        fn_test,
    )
    train = _load_mnist_vectors(fn_train)
    test = _load_mnist_vectors(fn_test)
    write_output(train, test, out_fn, "euclidean")


# Creates a 'deep image descriptor' dataset using the 'deep10M.fvecs' sample
# from http://sites.skoltech.ru/compvision/noimi/. The download logic is adapted
# from the script https://github.com/arbabenko/GNOIMI/blob/master/downloadDeep1B.py.
def deep_image(out_fn: str) -> None:
    yadisk_key = "https://yadi.sk/d/11eDCm7Dsn9GA"
    response = urlopen(
        "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key="
        + yadisk_key
        + "&path=/deep10M.fvecs"
    )
    response_body = response.read().decode("utf-8")

    dataset_url = response_body.split(",")[0][9:-1]
    filename = "data/deep-image.fvecs"
    # download(dataset_url, filename)

    # In the fvecs file format, each vector is stored by first writing its
    # length as an integer, then writing its components as floats.
    fv = numpy.fromfile(filename, dtype=numpy.float32)
    dim = fv.view(numpy.int32)[0]
    fv = fv.reshape(-1, dim + 1)[:, 1:]

    X_train, X_test = train_test_split(fv[:1000000])
    write_output(X_train, X_test, out_fn, "angular")


def transform_bag_of_words(filename: str, n_dimensions: int, out_fn: str) -> None:
    import gzip

    from scipy.sparse import lil_matrix
    from sklearn import random_projection
    from sklearn.feature_extraction.text import TfidfTransformer

    with gzip.open(filename, "rb") as f:
        file_content = f.readlines()
        entries = int(file_content[0])
        words = int(file_content[1])
        file_content = file_content[3:]  # strip first three entries
        print("building matrix...")
        A = lil_matrix((entries, words))
        for e in file_content:
            doc, word, cnt = [int(v) for v in e.strip().split()]
            A[doc - 1, word - 1] = cnt
        print("normalizing matrix entries with tfidf...")
        B = TfidfTransformer().fit_transform(A)
        print("reducing dimensionality...")
        C = random_projection.GaussianRandomProjection(n_components=n_dimensions).fit_transform(B)
        X_train, X_test = train_test_split(C)
        write_output(numpy.array(X_train), numpy.array(X_test), out_fn, "angular")


def nytimes(out_fn: str, n_dimensions: int) -> None:
    fn = "data/nytimes_%s.txt.gz" % n_dimensions
    download(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz", fn
    )  # noqa
    transform_bag_of_words(fn, n_dimensions, out_fn)


def random_float(out_fn: str, n_dims: int, n_samples: int, centers: int, distance: str) -> None:
    import sklearn.datasets

    X, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=centers, random_state=1)
    X_train, X_test = train_test_split(X, test_size=0.1)
    write_output(X_train, X_test, out_fn, distance)


def random_bitstring(out_fn: str, n_dims: int, n_samples: int, n_queries: int) -> None:
    import sklearn.datasets

    Y, _ = sklearn.datasets.make_blobs(n_samples=n_samples, n_features=n_dims, centers=n_queries, random_state=1)
    X = numpy.zeros((n_samples, n_dims), dtype=numpy.bool_)
    for i, vec in enumerate(Y):
        X[i] = numpy.array([v > 0 for v in vec], dtype=numpy.bool_)

    X_train, X_test = train_test_split(X, test_size=n_queries)
    write_output(X_train, X_test, out_fn, "hamming", "bit")


def sift_hamming(out_fn: str, fn: str) -> None:
    import tarfile

    local_fn = fn + ".tar.gz"
    url = "http://web.stanford.edu/~maxlam/word_vectors/compressed/%s/%s.tar.gz" % (path, fn)  # noqa
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        n_words, k = [int(z) for z in next(f).strip().split()]
        X = numpy.zeros((n_words, k), dtype=numpy.bool_)
        for i in range(n_words):
            X[i] = numpy.array([float(z) > 0 for z in next(f).strip().split()[1:]], dtype=numpy.bool_)

        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def sift_hamming(out_fn: str, fn: str) -> None:
    import tarfile

    local_fn = os.path.join("data", fn + ".tar.gz")
    url = "http://sss.projects.itu.dk/ann-benchmarks/datasets/%s.tar.gz" % fn
    download(url, local_fn)
    print("parsing vectors in %s..." % local_fn)
    with tarfile.open(local_fn, "r:gz") as t:
        f = t.extractfile(fn)
        lines = f.readlines()
        X = numpy.zeros((len(lines), 256), dtype=numpy.bool_)
        for i, line in enumerate(lines):
            X[i] = numpy.array([int(x) > 0 for x in line.decode().strip()], dtype=numpy.bool_)
        X_train, X_test = train_test_split(X, test_size=1000)
        write_output(X_train, X_test, out_fn, "hamming", "bit")


def lastfm(out_fn: str, n_dimensions: int, test_size: int = 50000) -> None:
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    import implicit
    from implicit.approximate_als import augment_inner_product_matrix
    from implicit.datasets.lastfm import get_lastfm

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = numpy.append(model.user_factors, numpy.zeros((model.user_factors.shape[0], 1)), axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, user_factors, out_fn, "angular")



def dbpedia_entities_openai_1M(out_fn, n = None):
    from sklearn.model_selection import train_test_split
    from datasets import load_dataset # huggingface datasets (pip install datasets)
    import numpy as np

    data = load_dataset("KShivendu/dbpedia-entities-openai-1M", split="train")
    if n is not None and n >= 100_000:
        data = data.select(range(n))

    embeddings = data.to_pandas()['openai'].to_numpy()
    embeddings = np.vstack(embeddings).reshape((-1, 1536))

    X_train, X_test = train_test_split(embeddings, test_size=10_000, random_state=42)

    write_output(X_train, X_test, out_fn, "angular")

def us_stock(out_fn):
    import zipfile
    from tqdm import tqdm
    import pandas as pd
    import os
    import numpy as np
    from data_collect.binary import fvecs_save
    from sklearn.model_selection import train_test_split

    local_fn = "data/us-stock-dataset.zip"
    url = 'https://www.kaggle.com/api/v1/datasets/download/footballjoe789/us-stock-dataset'
    download(url, local_fn)
    intervals = []
    summaries = []
    data_dir = 'data/us_stock'
    stock_price_date = '2024' # you can setup the date you interested

    with zipfile.ZipFile(local_fn, 'r') as zf:
        zf.extract('Stock_List.csv', data_dir)
        stocks = [file for file in zf.namelist() if file.startswith('Data/Stocks/')]
        for st in stocks:
            zf.extract(st, data_dir)
        stock_list = pd.read_csv(f'{data_dir}/Stock_List.csv')
        print("exract stock dataset...")
        for ticker in tqdm(stock_list['Symbol']):
            pth = f'{data_dir}/Data/Stocks/{ticker}.csv'
            if os.path.exists(pth):
                df = pd.read_csv(pth)
                df['Year'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z', utc=True).dt.year
                target_date = pd.to_datetime(stock_price_date, format='%Y').year
                filtered_df = df[df['Year'] == target_date][['Low', 'High', 'Date']]
                for i in range(len(filtered_df)):
                    low = filtered_df.iloc[i,0]
                    high = filtered_df.iloc[i,1]
                    dt = filtered_df.iloc[i,2]
                    # you can collect information for the stock yourself :)
                    # for simplicity, here, we only use the stock symbol
                    summary = f"{ticker}-{dt}" 
                    intervals.append([low, high])
                    summaries.append(summary)
    num_base = len(summaries)
    print(f'collect {num_base} stock entries')
    intervals = np.array(intervals, dtype=np.float32)
    # use 'sentence transformer' model for the stock vectorlization
    # paraphrase-MiniLM-L6-v2 model return 384 dimensions vectors
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    print('generate stock embeddings')
    embeddings = model.encode(summaries)
    X = np.concatenate([embeddings, intervals], axis=1, dtype=np.float32)
    # print(type(X))
    test_size = int(0.01 * num_base)
    X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
    fvecs_save("data/us-stock-384-euclidean.train.fvecs", X_train[:,:-2])
    fvecs_save("data/us-stock-384-euclidean.test.fvecs", X_test[:,:-2])
    fvecs_save("data/us-stock-384-euclidean..train.itv", X_train[:,-2:])
    fvecs_save("data/us-stock-384-euclidean..test.itv", X_test[:,-2:])

def covid19_epitope_prediction(out_fn):
    import zipfile
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.utils import resample

    # Average length of an amino acid in nanometers
    AA_TO_NM = 0.34

    local_fn = "data/covid19_epitope_prediction.zip"
    url = "https://www.kaggle.com/api/v1/datasets/download/futurecorporation/epitope-prediction"
    download(url, local_fn)
    data_dir = 'data/covid19_epitope_prediction'
    with zipfile.ZipFile(local_fn, 'r') as zf:
        zf.extract('input_bcell.csv', data_dir)
        zf.extract('input_sars.csv', data_dir)

    train = pd.read_csv(os.path.join(data_dir, 'input_bcell.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'input_sars.csv'))

    data_fields = [
        "parent_protein_id",    # parent protein ID
        "protein_seq",          # parent protein sequence
        "peptide_seq",          # peptide sequence
        "chou_fasman",          # peptide feature, β turn
        "emini",                # peptide feature, relative surface accessibility
        "kolaskar_tongaonkar",  # peptide feature, antigenicity
        "parker",               # peptide feature, hydrophobicity
        "isoelectric_point",    # protein feature
        "aromaticity",          # protein feature
        "hydrophobicity",       # protein feature
        "stability"             # protein feature
    ]

    # Convert amino acid positions to nanometer positions with random gaps for protein folding errors
    np.random.seed(42)  # For reproducibility
    gap_range = 1e5  # nm
    train['start_position_nm'] = train['start_position'] * AA_TO_NM + np.random.uniform(-gap_range, 0, size=len(train))
    train['end_position_nm'] = train['end_position'] * AA_TO_NM + np.random.uniform(0, gap_range, size=len(train))
    test['start_position_nm'] = test['start_position'] * AA_TO_NM + np.random.uniform(-gap_range, 0, size=len(test))
    test['end_position_nm'] = test['end_position'] * AA_TO_NM + np.random.uniform(0, gap_range, size=len(test))

    interval_fields = [
        "start_position_nm",   # start position of peptide in nm
        "end_position_nm",      # end position of peptide in nm
    ]

    # Function to encode protein sequences
    def encode_protein_sequences(sequences, k=3, dim=256):
        from sklearn.feature_extraction.text import HashingVectorizer
        vectorizer = HashingVectorizer(analyzer='char', ngram_range=(k, k), n_features=dim)
        encoded = vectorizer.transform(sequences)
        return encoded.toarray(), vectorizer

    # Encode protein sequences
    protein_encoder = None
    train_protein_encoded, protein_encoder = encode_protein_sequences(train['protein_seq'])
    test_protein_encoded = protein_encoder.transform(test['protein_seq']).toarray()

    # Encode peptide sequences
    train_peptide_encoded = protein_encoder.transform(train['peptide_seq']).toarray()
    test_peptide_encoded = protein_encoder.transform(test['peptide_seq']).toarray()

    # Get other numerical features
    train_other = train[data_fields[3:]]
    test_other = test[data_fields[3:]]

    # Combine all features
    train_data = np.concatenate([
        train_protein_encoded,
        train_peptide_encoded,
        train_other
    ], axis=1)
    test_data = np.concatenate([
        test_protein_encoded,
        test_peptide_encoded,
        test_other
    ], axis=1)

    train_itv = train[interval_fields].values
    test_itv = test[interval_fields].values

    # Scale numerical features
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Save the expanded data
    from data_collect.binary import fvecs_save
    print(train_data.shape, test_data.shape, train_itv.shape, test_itv.shape)
    fvecs_save("data/covid19-epitope-prediction-euclidean.train.fvecs", train_data)
    fvecs_save("data/covid19-epitope-prediction-euclidean.test.fvecs", test_data)
    fvecs_save("data/covid19-epitope-prediction-euclidean..train.itv", train_itv)
    fvecs_save("data/covid19-epitope-prediction-euclidean..test.itv", test_itv)
    

def ucf_crime(out_fn):
    import zipfile
    local_fn = "data/ucf-crime-dataset.zip"
    url = "https://www.kaggle.com/api/v1/datasets/download/odins0n/ucf-crime-dataset"
    data_dir = "data/ucf_crime"
    download(url, local_fn)

    with zipfile.ZipFile(local_fn, 'r') as zf:
        zf.extractall(data_dir)

    import os
    train_dir, test_dir = os.path.join(data_dir, 'Train'), os.path.join(data_dir, 'Test')
    
    def read_images(folder_path):
        png_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    png_files.append(file_path)
        return png_files

    train_images = read_images(train_dir)[:100_000]
    test_images = read_images(test_dir)[:10_000]

    import cv2
    import numpy as np
    from tqdm import tqdm

    def convert_images_to_vectors(png_files):
        vectors = []
        for file in tqdm(png_files):
            img = cv2.imread(file)
            img_array = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            vectors.append(img_array.flatten())
        return numpy.array(vectors)

    X_train = convert_images_to_vectors(train_images)
    X_test = convert_images_to_vectors(test_images)
    write_output(X_train, X_test, out_fn, "euclidean")


from .bigann_datasets import Text2Image1B


DATASETS: Dict[str, Callable[[str], None]] = {
    "deep-image-96-angular": deep_image,
    "fashion-mnist-784-euclidean": fashion_mnist,
    "gist-960-euclidean": gist,
    "glove-25-angular": lambda out_fn: glove(out_fn, 25),
    "glove-50-angular": lambda out_fn: glove(out_fn, 50),
    "glove-100-angular": lambda out_fn: glove(out_fn, 100),
    "glove-200-angular": lambda out_fn: glove(out_fn, 200),
    "mnist-784-euclidean": mnist,
    "random-xs-20-euclidean": lambda out_fn: random_float(out_fn, 20, 10000, 100, "euclidean"),
    "random-s-100-euclidean": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "euclidean"),
    "random-xs-20-angular": lambda out_fn: random_float(out_fn, 20, 10000, 100, "angular"),
    "random-s-100-angular": lambda out_fn: random_float(out_fn, 100, 100000, 1000, "angular"),
    "random-xs-16-hamming": lambda out_fn: random_bitstring(out_fn, 16, 10000, 100),
    "random-s-128-hamming": lambda out_fn: random_bitstring(out_fn, 128, 50000, 1000),
    "random-l-256-hamming": lambda out_fn: random_bitstring(out_fn, 256, 100000, 1000),
    "sift-128-euclidean": sift,
    "nytimes-256-angular": lambda out_fn: nytimes(out_fn, 256),
    "nytimes-16-angular": lambda out_fn: nytimes(out_fn, 16),
    "lastfm-64-dot": lambda out_fn: lastfm(out_fn, 64),
    "sift-256-hamming": lambda out_fn: sift_hamming(out_fn, "sift.hamming.256"),
    "us-stock-384-euclidean": us_stock,
    "ucf-crime-4096-euclidean": ucf_crime,
    "covid19-epitope-prediction-euclidean": covid19_epitope_prediction,
    "text2image-10M": Text2Image1B(10).prepare()
}

DATASETS.update({
    f"dbpedia-openai-{n//1000}k-angular": lambda out_fn, i=n: dbpedia_entities_openai_1M(out_fn, i)
    for n in range(100_000, 1_100_000, 100_000)
})