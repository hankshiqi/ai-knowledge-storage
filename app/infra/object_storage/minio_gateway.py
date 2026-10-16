from minio import Minio

from app.infra.config.providers import infra_config
import urllib.parse

from app.shared.clients.minio_utils import get_minio_client

class MinioGateway:
    @property
    def bucket_name(self) -> str:
        """获取 MinIO 存储桶名称（从全局配置读取）"""
        return infra_config.minio.bucket_name

    @property
    def image_dir(self) -> str:
        """获取 MinIO 中存放图片的目录路径（从全局配置读取）"""
        return infra_config.minio.minio_img_dir

    def client(self) -> Minio:
        return get_minio_client()

    def build_image_url(self, stem: str, object_name: str) -> str:
        """构建图片 URL"""
        scheme = "https" if infra_config.minio.minio_secure else "http"
        
        # 1. 处理 endpoint，防止用户手误写了协议头
        endpoint = infra_config.minio.endpoint.strip("/")
        if "://" in endpoint:
            endpoint = endpoint.split("://")[1]
            
        # 2. 对 object_name 做 URL 编码（处理中文、空格等特殊字符）
        # safe='' 表示连 '/' 也不作为安全字符（如果你希望保留路径结构，可改为 safe='/'）
        safe_object_name = urllib.parse.quote(object_name, safe='')
        
        # 3. 拼接路径
        path = "/".join(
            str(p).strip("/")
            for p in (self.bucket_name, self.image_dir, stem, safe_object_name)
            if p
        )
        
        return f"{scheme}://{endpoint}/{path}"


minio_gateway = MinioGateway()

if __name__ == "__main__":
    minio_gateway2 = MinioGateway()
    url = minio_gateway2.build_image_url(stem="user_01", object_name="生成的风景图.png")
    print(url)