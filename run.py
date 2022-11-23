from model import CausalBERT
from data import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import RichProgressBar
import argparse
from rich.console import Console
from rich.table import Table
import warnings 
warnings.filterwarnings("ignore")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str)
    parser.add_argument('--model_name', default='distilbert-base-uncased', type=str)
    parser.add_argument('--num_labels', default=2, type=int)
    parser.add_argument('--g_weight', default=0.1, type=float)
    parser.add_argument('--q_weight', default=0.1, type=float)
    parser.add_argument('--mlm_weight', default=1.0, type=float)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--accelerator', default='cpu', type=str)
    parser.add_argument('--data', default='testdata.csv', type=str)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--ckpt_path', default='last', type=str)
    args = parser.parse_args()

    seed_everything(args.seed)

    datamodule = DataModule(
        args.data,
        batch_size=args.batch_size, debug=True
    )

    model = CausalBERT(
        model_name=args.model_name,
        num_labels=args.num_labels,
        g_weight=args.g_weight,
        q_weight=args.q_weight,
        mlm_weight=args.mlm_weight,
    )

    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        callbacks=RichProgressBar()
    )

    if args.mode[0]=='fit':
        print('here')
        datamodule.setup('fit')
        trainer.fit(model=model, datamodule=datamodule)

    if args.mode[0]=='predict':
        datamodule.setup('predict')
        predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=args.ckpt_path)


