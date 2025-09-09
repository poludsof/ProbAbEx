import ProbAbEx as PAE



""" ========= Create and train model ========= """
ts = PAE.train_vaeac(epochs=20, lr=0.001f0, batch_size=100)



""" ================ Imputation and Sampling ================= """
x = PAE.load_binary_mnist_matrix()[:, 2]
mask = PAE.block_mask()
mask = PAE.random_mask(10; D=784)

x_img = PAE.sample_and_save_png(ts, x, mask; binary=true)

x_img2 = PAE.sample_and_save(x, mask, ts2, binary=true)



""" ================ Save and Load Model ================= """

PAE.save_vaeac_jls(ts, "models/mnist_vaeac_model_20.jls")
ts2 = PAE.load_vaeac_jls("models/mnist_vaeac_model_20.jls"; lr=0.001f0)


PAE.save_vaeac(ts, "models/mnist_vaeac_model_20.bson")
# ts2 = PAE.load_vaeac("models/mnist_vaeac_model.bson"; lr=learning_rate)
