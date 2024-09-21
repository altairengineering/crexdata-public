# Wayeb

Wayeb is a Complex Event Processing and Forecasting (CEP/F) engine written in [Scala](http://scala-lang.org).
It is based on symbolic automata and full- or variable-order Markov models.

## License

Copyright (c) Elias Alevizos

Wayeb comes with ABSOLUTELY NO WARRANTY.

Wayeb follows a dual licensing scheme.

For use by individuals in a line of work not supported/endorsed/promoted financially or otherwise by any public or
private legal entities,
Wayeb is licensed under the [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/).
To view a copy of this license, visit [http://creativecommons.org/licenses/by-nc-nd/4.0/](http://creativecommons.org/licenses/by-nc-nd/4.0/)
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

For commercial/institutional/governmental use or any other use by private or public
legal entities, sharing, modifying and distributing Wayeb or any derivatives of it
in any form, such as source code, libraries and executables, requires the written
permission of its author(s) (Elias Alevizos) and/or a possible request for licensing fees.
Wayeb's author(s) retain the right to deny the granting of such permission without providing any justification.
In case of any request(s), silence on the part of the author(s) implies refusal
to grant permission.

## Documentation

- [Building](docs/building.md)
- [Overview](docs/overview.md)
- [Recognition](docs/cep.md)
- [Forecasting with full-oder Markov models](docs/ceffmm.md)
- [Forecasting with variable-oder Markov models](docs/cefvmm.md)
- [Using Wayeb as a library](docs/lib.md)
- [How to cite Wayeb](docs/references.md)
- [Reproducing experimental results](docs/experiments.md)

## Reference
If you want to cite Wayeb, used the following references:

(Version that works only with classical automata and full-order Markov models)
```
@inproceedings{DBLP:conf/debs/AlevizosAP17,
  author    = {Elias Alevizos and
               Alexander Artikis and
               George Paliouras},
  title     = {Event Forecasting with Pattern Markov Chains},
  booktitle = {Proceedings of the 11th {ACM} International Conference on Distributed
               and Event-based Systems, {DEBS} 2017, Barcelona, Spain, June 19-23,
               2017},
  pages     = {146--157},
  publisher = {{ACM}},
  year      = {2017},
  url       = {https://doi.org/10.1145/3093742.3093920},
  doi       = {10.1145/3093742.3093920}
} 
```

(Version that works with symbolic automata and full-order Markov models)
```
@inproceedings{DBLP:conf/lpar/AlevizosAP18,
  author    = {Elias Alevizos and
               Alexander Artikis and
               Georgios Paliouras},
  editor    = {Gilles Barthe and
               Geoff Sutcliffe and
               Margus Veanes},
  title     = {Wayeb: a Tool for Complex Event Forecasting},
  booktitle = {{LPAR-22.} 22nd International Conference on Logic for Programming,
               Artificial Intelligence and Reasoning, Awassa, Ethiopia, 16-21 November
               2018},
  series    = {EPiC Series in Computing},
  volume    = {57},
  pages     = {26--35},
  publisher = {EasyChair},
  year      = {2018},
  url       = {https://easychair.org/publications/paper/VKP1}
}

```

(Version that works only with symbolic automata and variable-order Markov models)
```
@article{DBLP:journals/vldbj/AlevizosAP20,
  author    = {Elias Alevizos and
               Alexander Artikis and
               Georgios Paliouras},
  title     = {Complex Event Forecasting with Prediction Suffix Trees: a Formal Framework},
  journal   = {VLDBJ}
  year      = {2020}
}
```

## Contributors

* Elias Alevizos (main developer).
* Emmanouil Ntoulias (distributed version).
* Maria Petropanagiotaki (relaxed selection policies).

