from uuid import UUID

from sqlalchemy.future import select
from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.infrastructure.databases.exceptions import EntityNotFoundError
from cognee.modules.users.exceptions import PermissionDeniedError
from cognee.modules.users.models import (
    Role,
    Tenant,
    UserRole,
)
from cognee.modules.users.models.ACL import ACL


async def delete_role(role_id: UUID, owner_id: UUID):
    """
    Delete a role and its associations.

    Args:
        role_id: Id of the role to delete.
        owner_id: Id of the request owner.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not role:
            raise EntityNotFoundError(message="Role not found.")

        tenant = (
            (await session.execute(select(Tenant).where(Tenant.id == role.tenant_id)))
            .scalars()
            .first()
        )

        if tenant.owner_id != owner_id:
            raise PermissionDeniedError(
                message="User submitting request does not have permission to delete this role."
            )

        # Remove all user-role associations
        await session.execute(delete(UserRole).where(UserRole.role_id == role_id))

        # Remove all ACL entries for this role's principal
        await session.execute(delete(ACL).where(ACL.principal_id == role_id))

        # Delete the role (cascade will handle the principal)
        await session.execute(delete(Role).where(Role.id == role_id))

        await session.commit()
