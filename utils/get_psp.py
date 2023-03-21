import segmentation_models as sm
sm.set_framework('tf.keras')

def get_psp_model(classes,summary=True):
    activation='softmax'


    dice_loss = sm.losses.DiceLoss(class_weights=None) 
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    metrics = [sm.metrics.IOUScore(threshold=0.5,name='IoU'), sm.metrics.FScore(threshold=0.5,name='F1')]
    
    BACKBONE1 = 'inceptionresnetv2'
    preprocess_input1 = sm.get_preprocessing(BACKBONE1)

    psp_model = sm.PSPNet(BACKBONE1,classes=classes,input_shape=(720,720,1),encoder_weights=None,
                        activation=activation,psp_dropout=0.4)
    psp_model.compile('adam',loss=[total_loss] ,
                        metrics=metrics)
    if summary:
        psp_model.summary()
    return psp_model