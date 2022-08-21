/**
 * Bbox is a single bounding box
 * An image can have multiple boxes, it will be BBox[]
 * When storing boxes for multiple images, it will be BBox[][]
 */
export type BBox = number[];

export interface TaskImageInfo {
  objects: { bbox: BBox }[];
  url: string;
  maxAreaBBox?: {
    bbox: BBox[]; // selected bbox (the one with the largest area), length should be 1
  };
}

export interface ImageInfo {
  filename: string;
  objects: LabeledBBox[];
  url: string;
  maxAreaBBox?: {
    bbox: BBox[]; // selected bbox (the one with the largest area), length should be 1
    name: string;
  };
}

export interface LabeledBBox {
  bbox: BBox;
  name: string;
}
