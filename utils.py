from config import *


class ptype:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def gprint(content):
    print(
        ptype.BLUE + 
        ptype.BOLD + 
        '{}'.format(content) + 
        ptype.END
    )

def nprint(title, content):
    gprint(title)
    pprint(content)


def ensure_fresh_dir(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)


class FileCategory(enum.Enum):
    TRAIN_FEATURES = 'train_features'
    TEST_FEATURES = 'test_features'
    TEST_TARGETS = 'test_targets'
    UNCATEGORIZED = 'uncategorized'

category_filters = {
    FileCategory.TRAIN_FEATURES: lambda f: 'train' in f,
    FileCategory.TEST_FEATURES: lambda f: 'test' in f,
    FileCategory.TEST_TARGETS: lambda f: 'RUL' in f, 
}

def filter_directory(
    directory:str, 
    category_filters:Dict[FileCategory, Callable[[str], bool]], 
    excluded_files:List[str]=None
) -> Dict[FileCategory, List[str]]:
    # Get a list of all files in the directory, excluding specified files or types
    files_list = [f for f in os.listdir(directory) if not excluded_files or f not in excluded_files]
    nprint('Files List Name', files_list)
    # Filter files based on each filter in category_filters
    categorized_files = {category: [f for f in files_list if filter_func(f)] for category, filter_func in category_filters.items()}
    # Identify files that do not fall into any specified category
    categorized_files[FileCategory.UNCATEGORIZED] = [f for f in files_list if all(f not in files for files in categorized_files.values())]
    return categorized_files

def get_datasets_name(files_list:List[str]) -> List[str]:
    pattern = r"(train|test|rul|_|\.txt)"
    return [re.sub(pattern, '', file, flags=re.IGNORECASE) for file in files_list]

def get_dataset(name:str, files_list:List[str]) -> str:
    return [f for f in files_list if name in f][0]


def _extract_gdrive_id(gdrive_url:str) -> str:
    match = re.search(r'(?:/d/|id=)([\w-]+)', gdrive_url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid Google Drive URL format. Could not extract file ID.")
        

def download_from_gdrive(gdrive_url: str, destination_dir: str, file_name: str, file_type: str = 'zip') -> None:
    """
    Downloads a file from Google Drive and extracts it to the specified directory.

    Parameters:
    ----------
    file_id : str
        The unique ID of the file on Google Drive.
    destination_dir : str
        The directory where the file will be downloaded and extracted.
    file_name : str
        The name to give the downloaded file (without extension).
    file_type : str, optional
        The type of file to download and extract, default is 'zip'.
    
    Returns:
    -------
    None
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # get Google Drive file ID 
    gdrive_id = _extract_gdrive_id(gdrive_url)
    # get Google Drive url 
    url = f'https://drive.google.com/uc?id={gdrive_id}'
    
    file_path = os.path.join(destination_dir, f'{file_name}.{file_type}')
    
    # Download file
    print(f"Downloading {file_name}.{file_type} from Google Drive...")
    gdown.download(gdrive_url, file_path, quiet=False, fuzzy=True)
    
    # Extract file if necessary
    if file_type in ['zip', 'tar', 'gztar', 'bztar', 'xztar']:
        print(f"Extracting {file_name}.{file_type}...")
        shutil.unpack_archive(file_path, destination_dir, file_type)
        os.remove(file_path)  # Remove the archive after extraction
        print(f"Extraction complete: {destination_dir}")
    else:
        print(f"Download complete: {file_path}")


def _prepare_in_dirs(task:str) -> None:
    ensure_fresh_dir(in_dir)
    ensure_fresh_dir(dataset_dir)
    if task=='eda' or task=='preprocessing':
        download_from_gdrive(
            gdrive_url=dataset_url, 
            destination_dir=dataset_dir, 
            file_name=in_dataset_name
        )
    elif 'tuning' in task:
        download_from_gdrive(
            gdrive_url=preprocessed_dataset_url, 
            destination_dir=dataset_dir, 
            file_name=dataset_name
        )
    elif 'tuned' in task:
        download_from_gdrive(
            gdrive_url=tuned_cnn_model_url, 
            destination_dir=models_dir, 
            file_name=dataset_name
        )
        download_from_gdrive(
            gdrive_url=preprocessed_dataset_url, 
            destination_dir=dataset_dir, 
            file_name=dataset_name
        )
    elif 'training' in task:
        download_from_gdrive(
            gdrive_url=preprocessed_dataset_url, 
            destination_dir=dataset_dir, 
            file_name=dataset_name
        )
    elif 'testing' in task:
        download_from_gdrive(
            gdrive_url=preprocessed_dataset_url, 
            destination_dir=dataset_dir, 
            file_name=dataset_name
        )
        if task=='testing-cnn':
            download_from_gdrive(
                gdrive_url=cnn_model_url, 
                destination_dir=models_dir, 
                file_name=dataset_name
            )
        elif task=='testing-lstm':
            download_from_gdrive(
                gdrive_url=lstm_model_url, 
                destination_dir=models_dir, 
                file_name=dataset_name
            )
        elif task=='testing-gru':
            download_from_gdrive(
                gdrive_url=gru_model_url, 
                destination_dir=models_dir, 
                file_name=dataset_name
            )
    else:
        raise Exception('{} not valid.'.format(task))
    nprint('Input directory', in_dir)
    nprint('URL input dataset', dataset_url)
    nprint('Input dataset directory', dataset_dir)

def _prepare_out_dirs(task:str) -> None:
    ensure_fresh_dir(out_dir)
    nprint('Output directory', out_dir)
    if task=='eda':
        ensure_fresh_dir(out_dir)
        ensure_fresh_dir(out_data_dir)
        ensure_fresh_dir(out_models_dir)
        nprint('Output directory', out_dir)
        nprint('Output data directory', out_data_dir)
        nprint('Output models directory', out_models_dir)
    elif 'preprocessing' in task:
        ensure_fresh_dir(out_dir)
        ensure_fresh_dir(out_data_dir)
        nprint('Output directory', out_dir)
        nprint('Output data directory', out_data_dir)
    elif 'tuning' in task:
        ensure_fresh_dir(out_tuning_dir)
        nprint('Output tuning directory', out_tuning_dir)
    elif 'tuned' in task:
        ensure_fresh_dir(out_models_dir)
        nprint('Output models directory', out_models_dir)
    elif 'training' in task:
        ensure_fresh_dir(out_models_dir)
        nprint('Output models directory', out_models_dir)
    elif 'testing' in task:
        ensure_fresh_dir(out_evaluation_dir)
        nprint('Output evaluation directory', out_evaluation_dir)
    else:
        raise Exception('{} not valid.'.format(task))
    ensure_fresh_dir(out_plots_dir)
    nprint('Output plots directory', out_plots_dir)

def prepare_dirs(task:str) -> None:
    _prepare_in_dirs(task=task)
    _prepare_out_dirs(task=task)


def sanitize(filename):
    # defining substitutions for invalid characters
    substitutions = {
        '/': '_per_',
        '\\': '_',
        ':': '-',
        '*': '_',
        '?': '',
        '"': "'",
        '<': '',
        '>': '',
        '|': '_',
        '(': '',
        ')': '',
    }
    # replacing characters based on the substitutions
    for old_char, new_char in substitutions.items():
        filename = filename.replace(old_char, new_char)
    # removing any remaining invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    return filename.strip()
