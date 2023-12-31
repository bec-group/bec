import { belongsTo, hasMany, model, property } from '@loopback/repository';
import { Dataset, Event, DatasetWithRelations, Session, SessionWithRelations } from '.';
import { SciBecEntity } from './scibecentity.model';

@model()
export class Scan extends SciBecEntity {

  @property({
    type: 'string',
  })
  scanType: string;

  @property({
    type: 'object',
  })
  scanParameter: Object;

  @property({
    type: 'object',
  })
  userParameter: Object;

  @property({
    type: 'string',
  })
  queue: string;

  @property({
    type: 'string',
  })
  scanId: string;

  @property({
    type: 'string',
  })
  requestId: string;

  @property({
    type: 'string',
  })
  queueId: string;

  @property({
    type: 'string',
  })
  exitStatus: string;

  @property({
    type: 'number',
  })
  scanNumber?: number;

  @property({
    type: 'object',
  })
  metadata: Object;

  @property({
    type: 'object',
  })
  files?: Object;

  @belongsTo(() => Session,
    {}, //relation metadata goes in here
    {// property definition goes in here
      mongodb: { dataType: 'ObjectId' }
    })
  sessionId?: string;

  @belongsTo(() => Dataset,
    {}, //relation metadata goes in here
    {// property definition goes in here
      mongodb: { dataType: 'ObjectId' }
    })
  datasetId?: string;

  @hasMany(() => Event, { keyTo: 'scanId' })
  events?: Event[];

  constructor(data?: Partial<Scan>) {
    super(data);
  }
}

export interface ScanRelations {
  // describe navigational properties here
  session?: SessionWithRelations;
  datasets?: DatasetWithRelations[];
}

export type ScanWithRelations = Scan & ScanRelations;

