import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from load_css import local_css

local_css("style.css")

# Description, folder name, number of layers
models = [
    ['H3, Associative Recall', 'grokking_assoc_recall_h3', 2],
    ['Haar Wavelets From H3 (M=10), Associative Recall', 'grokking_assoc_recall_h3_wavelet_m10', 2],
    ['Haar Wavelets From H3 (M=5), Associative Recall', 'grokking_assoc_recall_h3_wavelet_m5', 2],
    ['S4 Simple, CIFAR', 'cifar_s4_simple', 3],
    ['Haar Wavelets Distilled from S4', 'haar_wavelets_s4_distilled', 3],
    ['SpaceTime Informer ETTh1', 'spacetime-etth-s=[336_2_2]-f=S-v=1-s=1-i=0', 3],
    ['SpaceTime Informer ETTh2', 'spacetime-etth-s=[336_2_2]-f=S-v=2-s=1-i=0', 3],
    ['SpaceTime Informer ETTm2', 'spacetime-ettm-s=[336_2_2]-f=S-v=2-s=1-i=0', 3],
]

st.write('# Visualize Convolutions')

cols = st.columns(2, gap='large')

query_params = st.experimental_get_query_params()

x = 0
layer_num = 0
show_frequency = False

if 'x' in query_params:
    x = int(query_params['x'][0])
    st.session_state.x = x
if 'layer_num' in query_params:
    layer_num = int(query_params['layer_num'][0])
    st.session_state.layer_num = layer_num
if 'show_frequency' in query_params:
    show_frequency = query_params['show_frequency'][0] == 'True'
    st.session_state.show_frequency = show_frequency

def set_query_params():
    st.experimental_set_query_params(x=st.session_state.x,
    layer_num=st.session_state.layer_num,
    show_frequency=st.session_state.show_frequency,)

x = cols[0].selectbox('Select Model', list(range(len(models))), format_func=lambda x: models[x][0], key='x', on_change=set_query_params)

layer_num = cols[1].selectbox('Select Layer', list(range(models[x][2])), key='layer_num', on_change=set_query_params)

show_frequency = st.checkbox('Show Frequency Response', key='show_frequency', on_change=set_query_params)

st.experimental_set_query_params(
    x=x,
    layer_num=layer_num,
    show_frequency=show_frequency,
)

file = f'data/{models[x][1]}/kernel_{layer_num}.npy'

kernel = np.load(file)


if show_frequency:
    kernel_fft = np.fft.rfft(kernel, axis=1, n=2 * kernel.shape[1])
    st.markdown('Legend: <span class="highlight blue">Time response</span>, <span class="highlight orange">Frequency real component</span>, <span class="highlight green">Frequency imaginary component</span>', unsafe_allow_html=True)

xmin, xmax = cols[0].slider('X Range', 0, kernel.shape[1], (0, min(64, kernel.shape[1])))
ymin = kernel.min().item()
ymax = kernel.max().item()

ymin, ymax = cols[1].slider('Y Range', ymin - (2.0 if show_frequency else 0.0), ymax + (2.0 if show_frequency else 0.0), (ymin, ymax))

with open(file, 'rb') as f:
    st.download_button('Download kernel', f, f'{models[x][1]}_kernel_{layer_num}.npy')


# fig, ax = plt.subplots(1, 2, figsize=(12, 4))

# im = ax[0].imshow(kernel)
# plt.colorbar(im, ax=ax[0])
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Hippo')


# ax[1].plot(kernel.T)
# ax[1].set_xlabel('Time')
# ax[1].set_ylabel('Conv Response')
# plt.show()
# st.pyplot(fig)


width = 4
chunk_size = 2
num_filters = kernel.shape[0]

st.write(f'### Visualizing {num_filters} filters')

num_chunks = num_filters // (chunk_size * width)

for chunk in range(num_chunks):
    offset = chunk * (chunk_size * width)
    fig, ax = plt.subplots(chunk_size, width, figsize = (12, 2 * chunk_size))
    for i in range(width):
        for j in range(chunk_size):
            ax[j, i].plot(kernel[width*j + i + offset], color='#4e79a7')
            if show_frequency:
                ax[j, i].plot(kernel_fft[width*j + i + offset].real, color='#f28e2b')
                ax[j, i].plot(kernel_fft[width*j + i + offset].imag, color='#59a14f')
            ax[j, i].set_ylim(ymin, ymax)
            ax[j, i].set_xlim(int(xmin) - 1, int(xmax) + 1)
            ax[j, i].grid()
    # plt.show()
    plt.close()

    st.pyplot(fig)