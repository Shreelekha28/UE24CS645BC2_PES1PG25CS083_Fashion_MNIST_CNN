import numpy as np

class Convolution:

    def __init__(self, num_filters, filter_size):

        self.num_filters = num_filters
        self.filter_size = filter_size

        # Random filter initialization
        self.filters = np.random.randn(
            num_filters,
            filter_size,
            filter_size
        ) / (filter_size * filter_size)

    def iterate_regions(self, image):

        h, w = image.shape

        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):

                region = image[
                    i:i + self.filter_size,
                    j:j + self.filter_size
                ]

                yield region, i, j

    def forward(self, input):

        self.last_input = input

        h, w = input.shape

        output = np.zeros((
            h - self.filter_size + 1,
            w - self.filter_size + 1,
            self.num_filters
        ))

        for region, i, j in self.iterate_regions(input):

            output[i, j] = np.sum(
                region * self.filters,
                axis=(1, 2)
            )

        return output