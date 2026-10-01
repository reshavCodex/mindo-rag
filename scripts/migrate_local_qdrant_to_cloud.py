from pathlib import Path
import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ENV_FILE = PROJECT_ROOT / "backend" / ".env"

load_dotenv(ENV_FILE)

LOCAL_QDRANT_PATH = PROJECT_ROOT / "data" / "vector_store" / "qdrant"
COLLECTION_NAME = "mindo_knowledge"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

BATCH_SIZE = 100


# ============================================================
# VALIDATION
# ============================================================

if not QDRANT_URL:
    raise RuntimeError(
        "QDRANT_URL is missing from the environment."
    )

if not QDRANT_API_KEY:
    raise RuntimeError(
        "QDRANT_API_KEY is missing from the environment."
    )

if not LOCAL_QDRANT_PATH.exists():
    raise RuntimeError(
        f"Local Qdrant storage was not found:\n{LOCAL_QDRANT_PATH}"
    )


# ============================================================
# CONNECT TO LOCAL QDRANT
# ============================================================

print("=" * 60)
print("MINDO QDRANT MIGRATION")
print("=" * 60)

print("\n[1/5] Connecting to local Qdrant...")
print(f"      Path: {LOCAL_QDRANT_PATH}")

local_client = QdrantClient(
    path=str(LOCAL_QDRANT_PATH)
)

print("      Local Qdrant connected.")


# ============================================================
# CONNECT TO QDRANT CLOUD
# ============================================================

print("\n[2/5] Connecting to Qdrant Cloud...")

cloud_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

print("      Qdrant Cloud connected.")


# ============================================================
# VERIFY COLLECTIONS
# ============================================================

print("\n[3/5] Verifying collections...")

local_collection = local_client.get_collection(
    collection_name=COLLECTION_NAME
)

cloud_collection = cloud_client.get_collection(
    collection_name=COLLECTION_NAME
)

local_vector_config = local_collection.config.params.vectors
cloud_vector_config = cloud_collection.config.params.vectors

print("\n      Local collection:")
print(f"        Name: {COLLECTION_NAME}")
print(f"        Vectors: {local_collection.points_count}")

print("\n      Cloud collection:")
print(f"        Name: {COLLECTION_NAME}")
print(f"        Vectors: {cloud_collection.points_count}")

# ------------------------------------------------------------
# Verify vector dimensions
# ------------------------------------------------------------

def get_vector_size(vector_config):
    if hasattr(vector_config, "size"):
        return vector_config.size

    if isinstance(vector_config, dict):
        first_config = next(iter(vector_config.values()))
        return first_config.size

    return None


def get_distance(vector_config):
    if hasattr(vector_config, "distance"):
        return vector_config.distance

    if isinstance(vector_config, dict):
        first_config = next(iter(vector_config.values()))
        return first_config.distance

    return None


local_vector_size = get_vector_size(local_vector_config)
cloud_vector_size = get_vector_size(cloud_vector_config)

local_distance = get_distance(local_vector_config)
cloud_distance = get_distance(cloud_vector_config)

print("\n      Vector configuration:")
print(f"        Local size:  {local_vector_size}")
print(f"        Cloud size:  {cloud_vector_size}")
print(f"        Local distance: {local_distance}")
print(f"        Cloud distance: {cloud_distance}")

if local_vector_size != cloud_vector_size:
    raise RuntimeError(
        "Vector dimension mismatch between local Qdrant and Qdrant Cloud."
    )

if str(local_distance) != str(cloud_distance):
    raise RuntimeError(
        "Vector distance mismatch between local Qdrant and Qdrant Cloud."
    )

local_count = local_collection.points_count

if local_count is None:
    raise RuntimeError(
        "Could not determine the number of local vectors."
    )

print(f"\n      Vectors to migrate: {local_count}")


# ============================================================
# MIGRATE POINTS
# ============================================================

print("\n[4/5] Migrating vectors...")

migrated = 0
offset = None

while True:

    points, next_offset = local_client.scroll(
        collection_name=COLLECTION_NAME,
        limit=BATCH_SIZE,
        offset=offset,
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        break

    cloud_points = []

    for point in points:

        if point.vector is None:
            raise RuntimeError(
                f"Point {point.id} has no vector."
            )

        cloud_points.append(
            models.PointStruct(
                id=point.id,
                vector=point.vector,
                payload=point.payload or {},
            )
        )

    cloud_client.upsert(
        collection_name=COLLECTION_NAME,
        points=cloud_points,
        wait=True,
    )

    migrated += len(cloud_points)

    print(
        f"      Migrated {migrated}/{local_count} vectors..."
    )

    if next_offset is None:
        break

    offset = next_offset


# ============================================================
# VERIFY MIGRATION
# ============================================================

print("\n[5/5] Verifying migration...")

cloud_count = cloud_client.count(
    collection_name=COLLECTION_NAME,
    exact=True,
).count

print(f"      Local vector count:  {local_count}")
print(f"      Cloud vector count:  {cloud_count}")

if cloud_count != local_count:
    raise RuntimeError(
        "Migration verification failed: "
        f"local={local_count}, cloud={cloud_count}"
    )


# ============================================================
# SAMPLE VERIFICATION
# ============================================================

sample_points, _ = local_client.scroll(
    collection_name=COLLECTION_NAME,
    limit=3,
    with_payload=True,
    with_vectors=True,
)

print("\n      Checking sample point IDs...")

for point in sample_points:

    result = cloud_client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=[point.id],
        with_payload=True,
        with_vectors=True,
    )

    if not result:
        raise RuntimeError(
            f"Sample point {point.id} was not found in Qdrant Cloud."
        )

    cloud_point = result[0]

    if cloud_point.id != point.id:
        raise RuntimeError(
            f"Point ID mismatch for {point.id}."
        )

    print(f"        OK: {point.id}")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 60)
print("MIGRATION COMPLETE")
print("=" * 60)

print(f"""
Collection:
    {COLLECTION_NAME}

Local vectors:
    {local_count}

Cloud vectors:
    {cloud_count}

Vector dimension:
    {cloud_vector_size}

Distance:
    {cloud_distance}

Status:
    SUCCESS

The local Qdrant database has NOT been modified or deleted.
""")

# Explicitly close clients.
local_client.close()
cloud_client.close()