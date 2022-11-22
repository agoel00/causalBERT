from model import CausalBERT
from data import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.cli import LightningCLI
import warnings 
import argparse

warnings.filterwarnings("ignore")
seed_everything(42)


data = DataModule('./testdata.csv', batch_size=8, debug=True)
# data.setup('fit')

model = CausalBERT(
    model_name='distilbert-base-uncased',
    num_labels=2,
    g_weight=0.1, q_weight=0.1,
    mlm_weight=1
)

# trainer = Trainer(
#     max_epochs=1,
#     accelerator='cpu',
#     devices=1,
#     detect_anomaly=True
# )
# trainer.fit(model, datamodule=data)

# data.setup('predict')
# predictions = trainer.predict(model, datamodule=data)

def cli_main():
    cli = LightningCLI(model, data)

if __name__=='__main__':
    cli_main()

# if __name__=='__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', default='fit')
#     parser.add_argument('--data', default='testdata.csv')
#     parser.add_argument('--g_weight', default=0.1)
#     parser.add_argument('--q_weight', default=0.1)
#     parser.add_argument('--mlm_weight', default=1.0)
#     parser.add_argument('--batch_size', default=8)
#     parser.add_argument('--bert', default='distilbert-base-uncased')
#     parser.add_argument('--num_labels', default=2)
#     parser.add_argument('--epochs', default=2)
#     parser.add_argument('--accelerator', default='cpu')
#     args = parser.parse_args()

#     data = DataModule(
#         args.data,
#         batch_size=args.batch_size,
#     )

#     model = CausalBERT(
#         model_name=args.bert,
#         num_labels=args.num_labels,
#         g_weight=args.g_weight,
#         q_weight=args.q_weight,
#         mlm_weight=args.mlm_weight
#     )

#     if args.mode=='fit':
#         data.setup('fit')
#         trainer = Trainer(
#             max_epochs=args.epochs,
#             accelerator=args.accelerator,
#         )
#         trainer.fit(model, datamodule=data)
#     elif args.mode=='predict':
#         data.setup('predict')

