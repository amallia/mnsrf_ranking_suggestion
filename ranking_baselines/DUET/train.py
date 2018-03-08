###############################################################################
# Author: Wasi Ahmad
# Project: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/10/wwwfp0192-mitra.pdf
# Date Created: 7/23/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, config, best_loss):
        self.model = model
        self.dictionary = dictionary
        self.config = config

        self.optimizer = optimizer
        self.best_dev_loss = best_loss
        self.times_no_improvement = 0
        self.stop = False
        self.train_losses = []
        self.dev_losses = []

    @staticmethod
    def compute_loss(logits, target):
        """
        Compute negative log-likelihood loss for a batch of predictions.
        :param logits: 2d tensor [batch_size x num_rel_docs_per_query]
        :param target: 2d tensor [batch_size x num_rel_docs_per_query]
        :return: average negative log-likelihood loss over the input mini-batch [autograd Variable]
        """
        loss = -(logits * target).sum(1)
        return loss.mean()

    def train_epochs(self, train_corpus, dev_corpus, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.train(train_corpus)
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                dev_loss = self.validate(dev_corpus)
                self.dev_losses.append(dev_loss)
                print('validation loss = %.4f' % dev_loss)
                # save model if dev loss goes down
                if self.best_dev_loss == -1 or self.best_dev_loss > dev_loss:
                    self.best_dev_loss = dev_loss
                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'best_loss': self.best_dev_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth.tar')
                    self.times_no_improvement = 0
                else:
                    self.times_no_improvement += 1
                    # no improvement in validation loss for last n iterations, so stop training
                    if self.times_no_improvement == 5:
                        self.stop = True
                # save the train and development loss plot
                helper.save_plot(self.train_losses, self.config.save_path, 'training', epoch + 1)
                helper.save_plot(self.dev_losses, self.config.save_path, 'dev', epoch + 1)
            else:
                break

    def train(self, train_corpus):
        # Turn on training mode which enables dropout.
        self.model.train()

        # splitting the data in batches
        train_batches = helper.batchify(train_corpus.data, self.config.batch_size)
        print('number of train batches = ', len(train_batches))

        start = time.time()
        print_loss_total = 0
        plot_loss_total = 0

        num_batches = len(train_batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_queries, train_docs, click_labels = helper.batch_to_tensor(train_batches[batch_no - 1],
                                                                             self.dictionary,
                                                                             self.config.max_query_length,
                                                                             self.config.max_doc_length)
            if self.config.cuda:
                # batch_size x max_query_length x vocab_size
                train_queries = train_queries.cuda()
                # batch_size x x num_rel_docs_per_query x max_doc_length x vocab_size
                train_docs = train_docs.cuda()
                # batch_size x num_rel_docs_per_query
                click_labels = click_labels.cuda()

            softmax_prob = self.model(train_queries, train_docs)
            loss = self.compute_loss(softmax_prob, click_labels)
            loss.backward()
            self.optimizer.step()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            if batch_no % self.config.print_every == 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.config.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.config.plot_every
                self.train_losses.append(plot_loss_avg)
                plot_loss_total = 0

    def validate(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('Number of dev batches = ', len(dev_batches))

        dev_loss = 0
        num_batches = len(dev_batches)
        for batch_no in range(1, num_batches + 1):
            dev_queries, dev_docs, click_labels = helper.batch_to_tensor(dev_batches[batch_no - 1], self.dictionary,
                                                                         self.config.max_query_length,
                                                                         self.config.max_doc_length)
            if self.config.cuda:
                dev_queries = dev_queries.cuda()
                dev_docs = dev_docs.cuda()
                click_labels = click_labels.cuda()

            softmax_prob = self.model(dev_queries, dev_docs)
            loss = self.compute_loss(softmax_prob, click_labels)
            dev_loss += loss.data[0]

        return dev_loss / num_batches
