# datajump.py
import numpy as np
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# Configurable: set your serial port or params here
def get_openbci_window(duration: float, fs: int) -> np.ndarray:
    """
    Streams 'duration' seconds of EEG data at 'fs' Hz from OpenBCI via BrainFlow.
    Returns a NumPy array shape (n_samples, n_channels).
    """
    # Initialize board
    params = BrainFlowInputParams()
    # e.g., params.serial_port = '/dev/ttyUSB0' or 'COM3'
    board = BoardShim(BoardIds.CYTON_BOARD.value, params)
    board.prepare_session()
    board.start_stream()

    # Wait for data
    BoardShim.log_message(BoardShim.INFO, 'Collecting data...')
    # Sleep to collect extra buffer
    import time; time.sleep(duration + 0.1)

    data = board.get_current_board_data(int((duration + 0.1) * fs))
    board.stop_stream()
    board.release_session()

    # EEG channels are typically channels 1–8 for Cyton
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    eeg_data = data[eeg_channels, :].T  # shape (n_samples, n_channels)
    return eeg_data

# Alias for main script
get_window = get_openbci_window
