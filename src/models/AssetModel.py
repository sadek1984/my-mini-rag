from .BaseDataModel import BaseDataModel
from .db_schemes import Asset
from .enums.DataBaseEnum import DataBaseEnum
from bson import ObjectId # Make sure ObjectId is imported from bson

class AssetModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        all_collections = await self.db_client.list_collection_names()
        if DataBaseEnum.COLLECTION_ASSET_NAME.value not in all_collections:
            self.collection = self.db_client[DataBaseEnum.COLLECTION_ASSET_NAME.value]
            indexes = Asset.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"], name=index["name"], 
                    unique=index["unique"]
                )
    
    async def create_asset(self, asset: Asset):
        """
        Creates an asset record in the database, ensuring correct data types.
        """
        # Convert the Pydantic model to a dictionary.
        asset_data = asset.dict(by_alias=True, exclude_unset=True)
        if "asset_project_id" in asset_data:
            asset_data["asset_project_id"] = ObjectId(asset_data["asset_project_id"])

        # Use the corrected asset_data dictionary for insertion.
        result = await self.collection.insert_one(asset_data)
        asset.id = result.inserted_id
        return asset

    async def get_all_project_assets(self, asset_project_id: str, asset_type: str):
        """
        Retrieves all assets for a given project, querying by ObjectId.
        """
        # The asset_project_id from the project model is an ObjectId, so the query
        # should be a direct match.
        records = await self.collection.find({
            "asset_project_id": ObjectId(asset_project_id) if isinstance(asset_project_id, str) else asset_project_id,
            "asset_type": asset_type
        }).to_list(length=None)
        return [
            Asset(**record)
            for record in records

        ]
    async def get_asset_record(self, asset_project_id: str, asset_name:str):
        record = await self.collection.find_one({
            "asset_project_id": ObjectId(asset_project_id) if isinstance(asset_project_id, str) else asset_project_id,
            "asset_name": asset_name
        })
        if record:
            return Asset(**record)
        return None