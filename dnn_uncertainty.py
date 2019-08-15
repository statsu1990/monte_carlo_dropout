import os

import numpy as np
import matplotlib.pyplot as plt

import cifar10_data
import classifier_cnn
import montecarlo_dropout

def Test_2class_sigmoid():
    """
    evaluation 2 class classification cnn with sigmoid activation.
    """

    # ####################################################
    # make data
    # ####################################################

    # dog and cat images in cifar10
    # dog label = 1, cat label = 0
    dog_cat = cifar10_data.Cifar10_Dog_Cat()
    dog_cat.make_binary_data()


    # ####################################################
    # make cnn model classifing dog and cat
    # 犬と猫を分類するCNN作成
    # ####################################################
    SAVED_MODEL_NAME = os.path.join(os.getcwd(),'saved_models_sigmoid','trained_model.h5')

    ## train model
    ## train acurracy 0.9227, loss 0.2634
    ## test  acurracy 0.8630, loss 0.3913
    
    do_training = True
    if do_training:
        cnn = classifier_cnn.BinaryClassifierCnn()
        cnn.built_model(dropout_rate=0.5, l2=0.0001)
        cnn.train_model(dog_cat.x_train, dog_cat.y_train, 
                        dog_cat.x_test, dog_cat.y_test, 
                        epochs=100, batch_size=64)
        cnn.save_model(save_file_name=SAVED_MODEL_NAME)

    ## load trained model
    cnn = classifier_cnn.BinaryClassifierCnn()
    cnn.load_model(SAVED_MODEL_NAME)


    # ####################################################
    # make monte carlo dropout model
    # モンテカルロドロップアウトを使うモデル作成
    # ####################################################
    md_cnn = montecarlo_dropout.MontecarloDropout()
    md_cnn.build_model(SAVED_MODEL_NAME)


    # ####################################################
    # predicted result
    # ####################################################
    MD_SAMPLING_NUM = 100
    # train, test, y
    y_train_cnn = cnn.model.predict(dog_cat.x_train)
    y_test_cnn = cnn.model.predict(dog_cat.x_test)
    y_train_mdcnn, std_train = md_cnn.md_predict(dog_cat.x_train, MD_SAMPLING_NUM)
    y_test_mdcnn, std_test = md_cnn.md_predict(dog_cat.x_test, MD_SAMPLING_NUM)
    # another label
    label_dict = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }
    ys_another_mdcnn = []
    stds_another = []
    for key in label_dict.keys():
        _y_mdcnn, _std = md_cnn.md_predict(cifar10_data.Cifar10_1Label(label=key).x_train, MD_SAMPLING_NUM)
        ys_another_mdcnn.append(_y_mdcnn.flatten())
        stds_another.append(_std.flatten())

    # save result dir
    SAVE_RESULT_DIR = os.path.join(os.getcwd(),'result_sigmoid')
    if not os.path.isdir(SAVE_RESULT_DIR):
        os.makedirs(SAVE_RESULT_DIR)


    # ####################################################
    # compare normal model with monte carlo dropout model
    # 比較
    # ####################################################

    # accuracy
    def calc_acc(_y, _pre_y):
        return 1 - np.average(np.logical_xor(_y > 0.5, _pre_y > 0.5))
    def print_calc_acc(_y_train, _pre_y_train, _y_test, _pre_y_test):
        print('  train acc, test acc : {0:.3f}, {1:.3f}'.format(
               calc_acc(_y_train, _pre_y_train), calc_acc(_y_test, _pre_y_test)))
    # normal cnn train acc, test acc : 0.923, 0.863
    print('\nnormal cnn')
    print_calc_acc(dog_cat.y_train, y_train_cnn, dog_cat.y_test, y_test_cnn)
    # md cnn train acc, test acc : 0.922, 0.863
    print('monte carlo dropout cnn')
    print_calc_acc(dog_cat.y_train, y_train_mdcnn, dog_cat.y_test, y_test_mdcnn)

    # predicted result md cnn vs cnn
    def plot_predicted_y_mdcnn_vs_cnn(_pre_y_mdcnn, _pre_y_cnn, _label, save_file_name, msize=None):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)

        maker_size = 10 if msize is None else msize

        _ax.scatter(_pre_y_mdcnn, _pre_y_cnn, s=maker_size, alpha=0.1, label=_label)
        _ax.plot([0,1], [0,1], color = "black")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.set_xlabel('Monte carlo dropout cnn predicted probability')
        _ax.set_ylabel('cnn predicted probability')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.legend()
        #plt.show()
        _fig.savefig(save_file_name)
        plt.clf()
    plot_predicted_y_mdcnn_vs_cnn(y_train_mdcnn, y_train_cnn, 'train', 
                                  os.path.join(SAVE_RESULT_DIR, 'predicted_prob_train.png'), None)
    plot_predicted_y_mdcnn_vs_cnn(y_test_mdcnn, y_test_cnn, 'test', 
                                  os.path.join(SAVE_RESULT_DIR, 'predicted_prob_test.png'), None)

    # ##########################
    # ROC and AUC of test data
    # ##########################
    def roc_and_auc(_y, _pre_y, _std, save_file_name):

        _posi_num = np.sum(_y > 0.5)
        _nega_num = np.sum(_y < 0.5)

        # tpf and fpr using y threshold
        _threshold_y = np.linspace(0, 1, num=100)
        _tpr1, _fpr1 = [], []
        for _thre_y in _threshold_y:
            _tpr1.append(np.sum(np.logical_and(_y > 0.5, _pre_y > _thre_y)) / _posi_num)
            _fpr1.append(np.sum(np.logical_and(_y < 0.5, _pre_y > _thre_y)) / _nega_num)

        # tpf and fpr using y + std threshold
        _threshold_a = np.tan(np.linspace(-pi*0.5+1e-6, pi*0.5-1e-6, num=100))
        _tpr2, _fpr2 = [], []
        for _thre_a in _threshold_a:
            _tpr2.append(np.sum(np.logical_and(_y > 0.5, _pre_y > 0.5 + _thre_a * _std)) / _posi_num)
            _fpr2.append(np.sum(np.logical_and(_y < 0.5, _pre_y > 0.5 + _thre_a * _std)) / _nega_num)

        # fig
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)
        _ax.plot(_fpr1, _tpr1, label='threshold = y')
        _ax.plot(_fpr2, _tpr2, label='threshold = 0.5 + a * std')
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.set_title('ROC curve')
        _ax.set_xlabel('FPR')
        _ax.set_ylabel('TPR')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.plot([0,1], [0,1], color = "black")
        _ax.legend()
        _fig.savefig(save_file_name)
        plt.clf()

        return

    roc_and_auc(dog_cat.y_test, y_test_mdcnn, std_test, os.path.join(SAVE_RESULT_DIR, 'roc_curve.png'))
    
    # ###############################
    # histgram of std of predicted y
    # ###############################
    def plot_std_histgram(_data_list, _label_list, save_dir):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(_data_list, label=_label_list, density=True, stacked=False)
        ax.set_title('Normalized histgram of monte carlo dropout std')
        ax.set_xlabel('Monte carlo dropout std')
        ax.set_ylabel('Normalized frequency')
        ax.legend()
        #plt.show()
        fig.savefig(save_dir)
        plt.clf()

    plot_std_histgram([std_train, std_test], ['train', 'test'], os.path.join(SAVE_RESULT_DIR, 'hist_std.png'))
    plot_std_histgram(stds_another, list(label_dict.values()), os.path.join(SAVE_RESULT_DIR, 'hist_std_another.png'))


    # ###############################
    # std vs predicted y
    # ###############################
    def plot_std_vs_predicted_y(_stds, _pre_ys, _labels, _max_std, save_file_name):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)

        for _std, _pre_y, _label in zip(_stds, _pre_ys, _labels):
            _ax.scatter(_std, _pre_y, s=10, alpha=0.1, label=_label)
        _ax.set_xlim(0, _max_std)
        _ax.set_title('monte carlo dropout std vs predicted probability')
        _ax.set_xlabel('Monte carlo dropout std')
        _ax.set_ylabel('Monte carlo dropout predicted probability')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.legend()
        #plt.show()
        _fig.savefig(save_file_name)
        plt.clf()

    max_std = 0
    max_std = np.maximum(np.max(std_train), np.max(std_test))
    for i in range(len(stds_another)):
        max_std = np.maximum(max_std, np.max(stds_another[i]))

    plot_std_vs_predicted_y([std_train], [y_train_mdcnn], ['train'], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_train.png'))
    plot_std_vs_predicted_y([std_test], [y_test_mdcnn], ['test'], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_test.png'))

    for _y_ano, _std_ano, _label in zip(ys_another_mdcnn, stds_another, list(label_dict.values())):
        plot_std_vs_predicted_y([_std_ano], [_y_ano], [_label], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_' + _label + '.png'))


    # ###############################
    # save picture
    # ###############################
    def save_pict(_x, _y, _pre_y_range, _pre_y, _std, _save_file_name):
        _in_range_idx = np.ravel(np.logical_and(_pre_y_range[0]<_pre_y, _pre_y_range[1]>_pre_y))

        _in_range_x = _x[_in_range_idx]
        _in_range_y = _y[_in_range_idx]
        _std_sort_idx = np.argsort(np.ravel(_std)[_in_range_idx])

        _use_num = 25

        # low std
        _low_std_pict = (_in_range_x[_std_sort_idx<_use_num] * 255 + 122.5).astype(np.uint8)
        _low_std_y = _in_range_y[_std_sort_idx<_use_num]

        _fig = plt.figure(figsize=(9, 8))
        _fig.suptitle('Low std pictures in predicted probability [' + str(_pre_y_range[0]) + ',' + str(_pre_y_range[1]) + ']')
        for _i, (_pict, _cor_y) in enumerate(zip(_low_std_pict, _low_std_y)):
            _ax = _fig.add_subplot(5, 5, _i+1)
            _ax.imshow(_pict)
            _ax.set_title('No.' + str(_i+1) + ',label ' + str(_cor_y[0]), fontsize=6, pad=1.0)
            _ax.axis('off')
        _fig.savefig(_save_file_name + '_low_std')
        plt.clf()

        # high std
        _high_std_pict = (_in_range_x[_std_sort_idx >= len(_std_sort_idx)-_use_num] * 255 + 122.5).astype(np.uint8)
        _high_std_y = _in_range_y[_std_sort_idx >= len(_std_sort_idx)-_use_num]

        _fig = plt.figure(figsize=(9, 8))
        _fig.suptitle('High std pictures in predicted probability [' + str(_pre_y_range[0]) + ',' + str(_pre_y_range[1]) + ']')
        for _i, (_pict, _cor_y) in enumerate(zip(_high_std_pict, _high_std_y)):
            _ax = _fig.add_subplot(5, 5, _i+1)
            _ax.imshow(_pict)
            _ax.set_title('No.' + str(_i+1) + ', label=' + str(_cor_y[0]), fontsize=6, pad=1.0)
            _ax.axis('off')
        _fig.savefig(_save_file_name + '_high_std')
        plt.clf()

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.48, 0.52], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict50'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.58, 0.62], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict60'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.68, 0.72], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict70'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.78, 0.82], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict80'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.88, 0.92], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict90'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.93, 0.97], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict95'), )

def Test_2class_softmax():
    """
    evaluation 2 class classification cnn with sigmoid activation.
    """

    # ####################################################
    # make data
    # ####################################################

    # dog and cat images in cifar10
    # dog label = 1, cat label = 0
    CLASS_NUM = 2
    dog_cat = cifar10_data.Cifar10_Dog_Cat()
    dog_cat.make_onehot_data()


    # ####################################################
    # make cnn model classifing dog and cat
    # 犬と猫を分類するCNN作成
    # ####################################################
    SAVED_MODEL_NAME = os.path.join(os.getcwd(),'saved_models_softmax','trained_model.h5')

    ## train model
    ## train acurracy 0.9200, loss 0.2546
    ## test  acurracy 0.8540, loss 0.3867
    
    do_training = True
    if do_training:
        cnn = classifier_cnn.MultiClassifierCnn(CLASS_NUM)
        cnn.built_model(dropout_rate=0.5, l2=0.0001)
        cnn.train_model(dog_cat.x_train, dog_cat.y_train, 
                        dog_cat.x_test, dog_cat.y_test, 
                        epochs=150, batch_size=64)
        cnn.save_model(save_file_name=SAVED_MODEL_NAME)

    ## load trained model
    cnn = classifier_cnn.MultiClassifierCnn(CLASS_NUM)
    cnn.load_model(SAVED_MODEL_NAME)


    # ####################################################
    # make monte carlo dropout model
    # モンテカルロドロップアウトを使うモデル作成
    # ####################################################
    md_cnn = montecarlo_dropout.MontecarloDropout()
    md_cnn.build_model(SAVED_MODEL_NAME)


    # ####################################################
    # predicted result
    # ####################################################
    MD_SAMPLING_NUM = 100
    # train, test, y
    y_train_cnn = cnn.model.predict(dog_cat.x_train)
    y_test_cnn = cnn.model.predict(dog_cat.x_test)
    y_train_mdcnn, std_train = md_cnn.md_predict(dog_cat.x_train, MD_SAMPLING_NUM)
    y_test_mdcnn, std_test = md_cnn.md_predict(dog_cat.x_test, MD_SAMPLING_NUM)
    #
    y_train_cnn = y_train_cnn[:,0]
    y_test_cnn = y_test_cnn[:,0]
    y_train_mdcnn, std_train = y_train_mdcnn[:,0], std_train[:,0]
    y_test_mdcnn, std_test = y_test_mdcnn[:,0], std_test[:,0]
    # another label
    label_dict = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }
    label_dict = {
        0 : 'airplane',
        1 : 'automobile',
    }
    ys_another_mdcnn = []
    stds_another = []
    for key in label_dict.keys():
        _y_mdcnn, _std = md_cnn.md_predict(cifar10_data.Cifar10_1Label(label=key).x_train, MD_SAMPLING_NUM)
        _y_mdcnn, _std = _y_mdcnn[:,0], _std[:,0]
        ys_another_mdcnn.append(_y_mdcnn.flatten())
        stds_another.append(_std.flatten())

    # save result dir
    SAVE_RESULT_DIR = os.path.join(os.getcwd(),'result_softmax')
    if not os.path.isdir(SAVE_RESULT_DIR):
        os.makedirs(SAVE_RESULT_DIR)

    # ####################################################
    # compare normal model with monte carlo dropout model
    # 比較
    # ####################################################
    # accuracy
    def calc_acc(_y, _pre_y):
        return 1 - np.average(np.logical_xor(_y > 0.5, _pre_y > 0.5))
    def print_calc_acc(_y_train, _pre_y_train, _y_test, _pre_y_test):
        print('  train acc, test acc : {0:.3f}, {1:.3f}'.format(
               calc_acc(_y_train, _pre_y_train), calc_acc(_y_test, _pre_y_test)))
    # normal cnn train acc, test acc : 0.920, 0.854
    print('\nnormal cnn')
    print_calc_acc(dog_cat.y_train[:,0], y_train_cnn, dog_cat.y_test[:,0], y_test_cnn)
    # md cnn train acc, test acc : 0.920, 0.855
    print('monte carlo dropout cnn')
    print_calc_acc(dog_cat.y_train[:,0], y_train_mdcnn, dog_cat.y_test[:,0], y_test_mdcnn)

    # predicted result md cnn vs cnn
    def plot_predicted_y_mdcnn_vs_cnn(_pre_y_mdcnn, _pre_y_cnn, _label, save_file_name, msize=None):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)

        maker_size = 10 if msize is None else msize

        _ax.scatter(_pre_y_mdcnn, _pre_y_cnn, s=maker_size, alpha=0.1, label=_label)
        _ax.plot([0,1], [0,1], color = "black")
        _ax.set_xlim(0, 1)
        _ax.set_ylim(0, 1)
        _ax.set_xlabel('Monte carlo dropout cnn predicted probability of dog')
        _ax.set_ylabel('cnn predicted probability of dog')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.legend()
        #plt.show()
        _fig.savefig(save_file_name)
        plt.clf()
    plot_predicted_y_mdcnn_vs_cnn(y_train_mdcnn, y_train_cnn, 'train', 
                                  os.path.join(SAVE_RESULT_DIR, 'predicted_prob_train.png'), None)
    plot_predicted_y_mdcnn_vs_cnn(y_test_mdcnn, y_test_cnn, 'test', 
                                  os.path.join(SAVE_RESULT_DIR, 'predicted_prob_test.png'), None)


    # ###############################
    # histgram of std of predicted y
    # ###############################
    def plot_std_histgram(_data_list, _label_list, save_dir):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(_data_list, label=_label_list, density=True, stacked=False)
        ax.set_title('Normalized histgram of monte carlo dropout std of dog')
        ax.set_xlabel('Monte carlo dropout std of dog')
        ax.set_ylabel('Normalized frequency')
        ax.legend()
        #plt.show()
        fig.savefig(save_dir)
        plt.clf()

    plot_std_histgram([std_train, std_test], ['train', 'test'], os.path.join(SAVE_RESULT_DIR, 'hist_std.png'))
    plot_std_histgram(stds_another, list(label_dict.values()), os.path.join(SAVE_RESULT_DIR, 'hist_std_another.png'))


    # ###############################
    # std vs predicted y
    # ###############################
    def plot_std_vs_predicted_y(_stds, _pre_ys, _labels, _max_std, save_file_name):
        _fig = plt.figure()
        _ax = _fig.add_subplot(111)

        for _std, _pre_y, _label in zip(_stds, _pre_ys, _labels):
            _ax.scatter(_std, _pre_y, s=10, alpha=0.1, label=_label)
        _ax.set_xlim(0, _max_std)
        _ax.set_title('monte carlo dropout std vs predicted probability of dog')
        _ax.set_xlabel('Monte carlo dropout std of dog')
        _ax.set_ylabel('Monte carlo dropout predicted probability of dog')
        _ax.grid(which='major',color='black',linestyle='-')
        _ax.legend()
        #plt.show()
        _fig.savefig(save_file_name)
        plt.clf()

    max_std = 0
    max_std = np.maximum(np.max(std_train), np.max(std_test))
    for i in range(len(stds_another)):
        max_std = np.maximum(max_std, np.max(stds_another[i]))

    plot_std_vs_predicted_y([std_train], [y_train_mdcnn], ['train'], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_train.png'))
    plot_std_vs_predicted_y([std_test], [y_test_mdcnn], ['test'], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_test.png'))

    for _y_ano, _std_ano, _label in zip(ys_another_mdcnn, stds_another, list(label_dict.values())):
        plot_std_vs_predicted_y([_std_ano], [_y_ano], [_label], max_std,
                            os.path.join(SAVE_RESULT_DIR, 'std_vs_prob_' + _label + '.png'))


    # ###############################
    # save picture
    # ###############################
    def save_pict(_x, _y, _pre_y_range, _pre_y, _std, _save_file_name):
        _in_range_idx = np.ravel(np.logical_and(_pre_y_range[0]<_pre_y, _pre_y_range[1]>_pre_y))

        _in_range_x = _x[_in_range_idx]
        _in_range_y = _y[_in_range_idx]
        _std_sort_idx = np.argsort(np.ravel(_std)[_in_range_idx])

        _use_num = 25

        # low std
        _low_std_pict = (_in_range_x[_std_sort_idx<_use_num] * 255 + 122.5).astype(np.uint8)
        _low_std_y = _in_range_y[_std_sort_idx<_use_num]

        _fig = plt.figure(figsize=(9, 8))
        _fig.suptitle('Low std pictures in predicted probability [' + str(_pre_y_range[0]) + ',' + str(_pre_y_range[1]) + ']')
        for _i, (_pict, _cor_y) in enumerate(zip(_low_std_pict, _low_std_y)):
            _ax = _fig.add_subplot(5, 5, _i+1)
            _ax.imshow(_pict)
            _ax.set_title('No.' + str(_i+1) + ',label ' + str(_cor_y[0]), fontsize=6, pad=1.0)
            _ax.axis('off')
        _fig.savefig(_save_file_name + '_low_std')
        plt.clf()

        # high std
        _high_std_pict = (_in_range_x[_std_sort_idx >= len(_std_sort_idx)-_use_num] * 255 + 122.5).astype(np.uint8)
        _high_std_y = _in_range_y[_std_sort_idx >= len(_std_sort_idx)-_use_num]

        _fig = plt.figure(figsize=(9, 8))
        _fig.suptitle('High std pictures in predicted probability [' + str(_pre_y_range[0]) + ',' + str(_pre_y_range[1]) + ']')
        for _i, (_pict, _cor_y) in enumerate(zip(_high_std_pict, _high_std_y)):
            _ax = _fig.add_subplot(5, 5, _i+1)
            _ax.imshow(_pict)
            _ax.set_title('No.' + str(_i+1) + ', label=' + str(_cor_y[0]), fontsize=6, pad=1.0)
            _ax.axis('off')
        _fig.savefig(_save_file_name + '_high_std')
        plt.clf()

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.48, 0.52], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict50'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.58, 0.62], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict60'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.68, 0.72], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict70'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.78, 0.82], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict80'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.88, 0.92], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict90'), )

    save_pict(dog_cat.x_train, dog_cat.y_train, [0.93, 0.97], y_train_mdcnn, std_train, 
              os.path.join(SAVE_RESULT_DIR, 'pict95'), )

# run uncertainty estimation test
Test_2class_sigmoid()
Test_2class_softmax()
