#################################DEPRECATED######################################

for lr, hr in train_ds.take(1):
        lr_batch = tf.cast(lr, tf.float32)
        sr_batch = model(lr_batch)
        sr_batch = tf.clip_by_value(sr_batch, 0, 255)
        sr_batch = tf.round(sr_batch)
        sr_batch = tf.cast(sr_batch, tf.uint8)
        print(sr_batch)





for lr, hr in train_ds:
    plt.imshow(lr[0], cmap=plt.cm.binary)
    plt.xlabel("pace")
    plt.show()

    plt.imshow(hr[0], cmap=plt.cm.binary)
    plt.xlabel("bolji pace")
    plt.show()




def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.from_tensor_slices((self.lr_dataset(), self.hr_dataset()))

        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=2), num_parallel_calls=AUTOTUNE)

        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds


def dataset(self, batch_size=16, repeat_count=None, random_transform=True):
        ds = tf.data.Dataset.zip((self.lr_dataset(), self.hr_dataset()))
        if random_transform:
            ds = ds.map(lambda lr, hr: random_crop(lr, hr, scale=2), num_parallel_calls=AUTOTUNE)
            # ds = ds.map(random_rotate, num_parallel_calls=AUTOTUNE)
            # ds = ds.map(random_flip, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(batch_size)
        ds = ds.repeat(repeat_count)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

download_archive("DIV2K_train_LR_bicubic_X2.zip", "./");
download_archive("DIV2K_valid_LR_bicubic_X2.zip", "./");
download_archive("DIV2K_train_HR.zip", "./");
download_archive("DIV2K_valid_HR.zip", "./");







epochs = 5
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_ds.take(400)):

        # plt.imshow(x_batch_train[0], cmap=plt.cm.binary)
        # plt.xlabel("pace")
        # plt.show()

        # plt.imshow(y_batch_train[0], cmap=plt.cm.binary)
        # plt.xlabel("bolji pace")
        # plt.show()


        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            # print(x_batch_train.shape)
            # print(y_batch_train.shape)

            logits = model(x_batch_train, training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * 64))


model.save("pace2.h5")

for lr, hr in valid_ds.take(1):
    predictions = model.predict(lr)

    plt.imshow(lr[0], cmap=plt.cm.binary)
    plt.xlabel("pace")
    plt.show()

    plt.imshow(predictions[0], cmap=plt.cm.binary)
    plt.xlabel("pace")
    plt.show()


#PLOT
fig = plt.figure(figsize=(16,16))

columns = 4
rows = 4

i = 0
c = 0


for lr, hr in valid_ds.take(40):

    if(c > 0 and c%4 == 0):
        i = 0
        plt.show()
        fig = plt.figure(figsize=(16,16))
   
    c+=1 
    
    i+=1
    fig.add_subplot(rows, columns, i)
    plt.imshow(lr[0], cmap=plt.cm.binary)
    

    i+=1
    fig.add_subplot(rows, columns, i)
    predictions = model.predict(norm(lr))
    
    x=[denorm(y) for y in predictions[0]]
    plt.imshow(x, cmap=plt.cm.binary)
    
    i+=1
    fig.add_subplot(rows, columns, i)
    x=[norm(y) for y in lr[0]]

    rez = cv2.resize(np.float32(x), (0,0), fx=2, fy=2)

    plt.imshow(rez, cmap=plt.cm.binary)


    i+=1
    fig.add_subplot(rows, columns, i)
    plt.imshow(hr[0], cmap=plt.cm.binary)


#OPTIMIZERS AND STUFF
def psnr(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true + 1e-8), axis=-1)))) / 2.303


def psnr_abs(y_true,y_pred):
    max_pixel = norm(255)
    return -(10.0 * K.log((max_pixel ** 2) / (K.mean(K.abs(y_pred - y_true + 1e-8), axis=-1)))) / 2.303


def norm(x):
    return ((tf.cast(x,dtype="float32")-tf.ones(x.shape, dtype="float32")*127.5)/255)

def denorm(x):
    return tf.cast((tf.cast(x,dtype="float32")*255+tf.ones(x.shape, dtype="float32")*127.5), dtype="int8")




optimizer = keras.optimizers.Nadam(learning_rate=n/10) #smanji lr
optimizer = keras.optimizers.SGD(learning_rate=1e-3)

loss_fn = keras.losses.MeanSquaredError()
loss_fn = psnr
loss_fn = keras.losses.MeanAbsoluteError()
loss_fn = psnr_abs



model = keras.Model(Input_img, decoded[0:,1:-1,1:-1])