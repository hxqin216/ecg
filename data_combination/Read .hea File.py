from IPython.display import display
import matplotlib.pyplot as plt
import wfdb

record = wfdb.rdheader('mit-bih-arrhythmia-database-1.0.0/101')
display(record.__dict__)


# Now you can access header information
# num_channels = header.sig_len
# sample_frequency = header.fs
# record_length = header.sig_len / header.fs



