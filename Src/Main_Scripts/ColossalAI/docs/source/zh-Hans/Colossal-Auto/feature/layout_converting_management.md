sharding specLayout Conversion
sharding spec
Device MeshN-d tensorsharding specX0X1...Xn-1i0n-1sharding spec
  1. source specone-step transform sharding specs
  2. one-step transform sharding specssharding specsource sharding specsharding spectransform pathone-step transform sharding spectarget sharding specsharding spec
  3. ab

| Source/target sharding spec pairs |All gather | Shard | All to All | One step transform | Best sharding spec |Transform path|
| :-:         | :-:              | :-:                  | :-:                       | :-:                     | :-:                     |:-:                     |
| $S_{01}RR RS_{01}R$  | $S_0RR$       | -           | $S_0RS_1, S_0S_1R$             | $S_0RR, S_0RS_1, S_0S_1R$             | $S_0RR$ | $S_0RR$
| $S_0RR, RS_{01}RR$  | $RRR$       | $S_0S_1R, S_0RS_1$           | $RS_0R, RRS_0$             | $RRR$, $S_0S_1R$, $S_0RS_1$, $RS_0R$, $RRS_0$             | $RS_0R$ | $S_0RR$ -> $RS_0R$
| $RS_0R, RS_{01}RR$  | $RRR$       | $RS_{01}R, S_1S_0R, RS_0S_1$           | $S_0RR, RRS_0$             | $RRR$, $RS_{01}R$, $S_1S_0R$, $RS_0S_1$, $S_0RR$, $RRS_0$             | $RS_{01}R$ | $S_0RR$ -> $RS_0R$ -> $RS_{01}R$
