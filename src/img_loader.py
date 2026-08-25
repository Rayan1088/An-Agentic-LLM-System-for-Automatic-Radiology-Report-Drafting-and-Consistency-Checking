import sys
from PIL import Image
from pathlib import Path
from src.logger import logging
from src.exception import CustomException
from typing import List,Optional, Any

logger= logging.getLogger("image_loader")

class ImageLoader:
    
    def __init__(self, 
                image_paths: Optional[List[str]]=None,
                image_dir: Optional[str]=None,
                uploaded_files: Optional[List[Any]]=None):
        
        image_ext={".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

        try:
            sources_given=sum(bool(x) for x in (image_paths, image_dir, uploaded_files))
            if sources_given==0:
                raise CustomException("Must provide one of: image_paths, image_dir, uploaded_files", sys)
            if sources_given>1:
                raise CustomException("Provide only ONE of: image_paths / image_dir / uploaded_files, not multiple", sys)
            
            self.mode: Optional[str] = None
            self.image_paths: List[Path]=[]
            self.uploaded_files: List[Any]=[]     
            
            if uploaded_files:
                files=uploaded_files if isinstance(uploaded_files, (list, tuple)) else [uploaded_files]     
            
                if not files:
                    raise CustomException("uploaded_files is empty", sys)
                
                self.uploaded_files=list(files)
                self.mode="uploaded"
                logger.info(f"ImageLoader initialized with {len(self.uploaded_files)} uploaded files.")  
                return

            if image_paths:
                paths=[Path(p) for p in image_paths]
            else:
                image_dir_path=Path(image_dir)
                if not image_dir_path.exists():
                    raise CustomException(f"Image directory not found: {image_dir}", sys)
                paths=sorted(p for p in image_dir_path.iterdir() if p.suffix.lower() in image_ext)             
            
            existing, missing= [],[]
            for p in paths:
                (existing if p.exists() else missing).append(p)
            if missing:
                logger.warning(f"{len(missing)} image paths not found on directory")  
            if not existing:
                raise CustomException("No valid images found provided paths", sys)
            
            self.image_paths=existing
            self.mode="paths"
            logger.info(f"ImageLoader initialized with {len(self.image_paths)} valid images")
        except CustomException as e:
            logger.error(f"Error occurred during load the images from directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Error occurred during load the images from directory: {e}")
            raise CustomException(e, sys)
    
    def load_raw_images(self):
        filenames, images=[], []
        if self.mode=="uploaded":
            for f in self.uploaded_files:
                try:
                    f.seek(0)
                    img=Image.open(f).convert("RGB")
                    filenames.append(getattr(f, "name", f"uploaded_{len(filenames)}.png"))
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Skipping unreadable uploaded file: {getattr(f, 'name', '<unknown>')}: {e}")
        else:
            ### if mode=="paths"
            for path in self.image_paths:
                try:
                    img=Image.open(path).convert("RGB")
                    filenames.append(str(path))
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Skipping unreadable image [{path}]: {e}")

        if not images:
            raise CustomException("No images are loaded", sys)
        
        logger.info(f"Loaded {len(images)} raw RGB images")
        return filenames, images
    

        