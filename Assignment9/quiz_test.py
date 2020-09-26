class QuizModel(BaseModel):
    def __init__(self, dropout_value=0.25):
        self.dropout_value = dropout_value

        super(QuizModel, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(1, 1), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.pool2 = nn.MaxPool2d(2, 2)

        self.convblock7 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1), dilation=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )
        self.convblock9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.convblock10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout_value)
        )

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.convblock11 = nn.Sequential(
            nn.Conv2d(128, 10, kernel_size=(1, 1), bias=False),
        )

    def forward(self, x):
        x1 = self.convblock1(x)  # x1 = Input
        x2 = self.convblock2(x1)  # x2 = x1
        x3 = self.convblock3(x1 + x2)  # x3 = x1 + x2
        x4 = self.pool1(x1 + x2 + x3)

        x4 = self.convblock4(x4)

        x5 = self.convblock5(x4)
        x6 = self.convblock6(x4 + x5)
        x7 = self.convblock6(x4 + x5 + x6)
        x8 = self.pool2(x5 + x6 + x7)

        x8 = self.convblock7(x8)

        x9 = self.convblock8(x8)
        x10 = self.convblock9(x8 + x9)
        x11 = self.convblock10(x8 + x9 + x10)

        x12 = self.gap(x11)
        x13 = self.convblock11(x12)

        x13 = x13.view(-1, 10)
        return x13