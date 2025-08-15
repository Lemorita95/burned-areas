from scripts.helpers import tf, torch, nn, PNG_RESOLUTION

class Model():

    def __init__(self):

        # Create a convolutional neural network
        model = tf.keras.models.Sequential([

            # input layer
            tf.keras.Input(shape=PNG_RESOLUTION),

            # Convolutional layer 1
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),


            # Convolutional layer 2
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),


            # Convolutional layer 3
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 4
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 5
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # Convolutional layer 6
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding='same'),
            tf.keras.layers.UpSampling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),

            # output layer
            tf.keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding='same'),
        ])

        # Train neural network
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.model = model
        

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=480, patch_size=16, in_channels=3, embed_dim=512):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, H, W, C = x.shape
        # unfold patches
        patches = x.permute(0, 3, 1, 2).unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)  # [B, C, nH, nW, pH, pW]
        patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, pH, pW]
        patches = patches.flatten(3)  # flatten patch spatial dims: [B, num_patches, C, pH*pW]
        patches = patches.flatten(2)  # flatten channels + pixels: [B, num_patches, C*pH*pW]

        embeddings = self.proj(patches)  # [B, num_patches, embed_dim]
        return embeddings
    

class TransformerModel(nn.Module):
    def __init__(self, img_size=480, patch_size=16, in_channels=3, embed_dim=512,
                 num_layers=6, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches_side = img_size // patch_size  # g
        self.num_patches = self.num_patches_side ** 2   # total patches

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.patch_embed.num_patches, embed_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project patch embeddings to pixels
        self.patch_to_pixels = nn.Linear(embed_dim, patch_size * patch_size)

    def forward(self, x):
        """
        x: [B, H, W, C]
        returns: [B, 1, H, W] logits
        """
        B = x.size(0)
        p = self.patch_size
        g = self.num_patches_side

        # 1. Patch embeddings
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # 2. Positional encoding
        x = x + self.pos_embed

        # 3. Transformer
        x = self.transformer(x)  # [B, num_patches, embed_dim]

        # 4. Project each patch to patch pixels
        x = self.patch_to_pixels(x)  # [B, num_patches, p*p]
        x = x.view(B, g, g, p, p)    # [B, gH, gW, pH, pW]

        # 5. Stitch patches back to full image
        x = x.permute(0, 1, 3, 2, 4)  # [B, gH, pH, gW, pW]
        x = x.reshape(B, g*p, g*p, 1)  # [B, H, W, 1]

        return x  # logits for BCEWithLogitsLoss