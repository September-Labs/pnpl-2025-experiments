import torch
import numpy as np 
import pywt

class DatasetWrapper:
    """
    A base class for wrapping datasets (e.g., PyTorch Datasets).
    
    Subclasses should override the `__getitem__` method to implement the 
    desired data transformation.

    Any attributes or methods not defined on the wrapper (e.g., helper
    methods like `generate_submission_in_csv`) will be automatically 
    forwarded to the wrapped dataset via `__getattr__`.
    """
    def __init__(self, dataset):
        """
        Initializes the wrapper.

        Args:
            dataset: The dataset to wrap.
        """
        self._dataset = dataset

    @property
    def dataset(self):
        """
        Provides read-only access to the wrapped dataset.
        """
        return self._dataset

    def __len__(self):
        """
        Returns the length of the wrapped dataset.
        """
        return len(self._dataset)

    def __getitem__(self, idx):
        """
        Gets an item from the wrapped dataset and applies a transformation.

        This method MUST be overridden by all subclasses.
        """
        # Example of what a subclass would do:
        # 1. Get the original data
        # data = self._dataset[idx]
        #
        # 2. Apply some transformation
        # transformed_data = self.my_transform(data)
        #
        # 3. Return the transformed data
        # return transformed_data
        
        raise NotImplementedError(
            f"{type(self).__name__} must implement the `__getitem__` method."
        )

    def __getattr__(self, name):
        """
        Forwards attribute/method access to the wrapped dataset if the 
        attribute is not found on the wrapper itself.

        This allows wrappers to be transparent for methods (like
        `generate_submission_in_csv`) or attributes defined on the 
        original dataset.
        """
        # Check if the attribute exists on the wrapped dataset
        if hasattr(self._dataset, name):
            # Return the attribute from the wrapped dataset
            return getattr(self._dataset, name)
        else:
            # If not found on wrapper or dataset, raise the standard error
            raise AttributeError(
                f"'{type(self).__name__}' object and its wrapped 'dataset' "
                f"have no attribute '{name}'"
            )

class DenoisedDatasetWrapper(DatasetWrapper):
    """
    Wrapper around LibriBrainCompetitionHoldout that applies wavelet denoising.
    """
    def __init__(self, dataset, wavelet='db4', level=3, threshold_type='soft', 
                 denoise_percentage=1.0, preserve_scale=True):
        """
        Args:
            dataset: Original LibriBrainCompetitionHoldout dataset
            wavelet: Wavelet for denoising
            level: Decomposition level
            threshold_type: 'soft' or 'hard' thresholding
            denoise_percentage: Fraction of threshold to apply (0-1), lower = less denoising
            preserve_scale: Whether to preserve original signal scale after denoising
        """
        self.dataset = dataset
        self.wavelet = wavelet
        self.level = level
        self.threshold_type = threshold_type
        self.denoise_percentage = denoise_percentage
        self.preserve_scale = preserve_scale
        
     
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original data
        data = self.dataset[idx]  # Shape: (306, 125) for MEG channels x timepoints
        
        # Convert to numpy for processing
        if torch.is_tensor(data):
            data_np = data.numpy()
            return_tensor = True
        else:
            data_np = data
            return_tensor = False
        
        # Apply wavelet denoising channel-wise
        denoised_data = np.zeros_like(data_np)
        
        for ch_idx in range(data_np.shape[0]):
            channel_data = data_np[ch_idx, :]
            
            # Store original statistics if preserving scale
            if self.preserve_scale:
                orig_mean = channel_data.mean()
                orig_std = channel_data.std()
            
            # Apply wavelet denoising
            coeffs = pywt.wavedec(channel_data, self.wavelet, level=self.level)
            
            # Estimate noise sigma using MAD
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            
            # Calculate threshold
            threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))
            threshold *= self.denoise_percentage  # Apply percentage
            
            # Threshold coefficients
            coeffs_thresh = list(coeffs)
            for j in range(1, len(coeffs)):
                if self.threshold_type == 'soft':
                    coeffs_thresh[j] = pywt.threshold(coeffs[j], threshold, mode='soft')
                else:
                    coeffs_thresh[j] = pywt.threshold(coeffs[j], threshold, mode='hard')
            
            # Reconstruct
            denoised_channel = pywt.waverec(coeffs_thresh, self.wavelet)[:len(channel_data)]
            
            # Restore scale if requested
            if self.preserve_scale and orig_std > 0:
                denoised_channel = (denoised_channel - denoised_channel.mean()) / denoised_channel.std()
                denoised_channel = denoised_channel * orig_std + orig_mean
            
            denoised_data[ch_idx, :] = denoised_channel
        
        # Convert back to tensor if needed
        if return_tensor:
            denoised_data = torch.from_numpy(denoised_data).float()
        
        return denoised_data

    def generate_submission_in_csv(self, predictions, output_path):
        """
        Delegate to the original dataset's method.
        The predictions are already from denoised data, so we just need
        the original dataset's metadata for creating the CSV.
        """
        return self.dataset.generate_submission_in_csv(predictions, output_path)
