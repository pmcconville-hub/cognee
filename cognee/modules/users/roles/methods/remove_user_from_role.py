from uuid import UUID

from sqlalchemy.future import select
from sqlalchemy import delete

from cognee.infrastructure.databases.relational import get_relational_engine
from cognee.modules.users.exceptions import (
    UserNotFoundError,
    RoleNotFoundError,
    PermissionDeniedError,
)
from cognee.modules.users.models import (
    User,
    Role,
    Tenant,
    UserRole,
)


async def remove_user_from_role(user_id: UUID, role_id: UUID, owner_id: UUID):
    """
    Remove a user from a role.

    Args:
        user_id: Id of the user.
        role_id: Id of the role.
        owner_id: Id of the request owner.
    """
    db_engine = get_relational_engine()
    async with db_engine.get_async_session() as session:
        user = (await session.execute(select(User).where(User.id == user_id))).scalars().first()
        role = (await session.execute(select(Role).where(Role.id == role_id))).scalars().first()

        if not user:
            raise UserNotFoundError
        if not role:
            raise RoleNotFoundError

        tenant = (
            (await session.execute(select(Tenant).where(Tenant.id == role.tenant_id)))
            .scalars()
            .first()
        )

        if tenant.owner_id != owner_id:
            raise PermissionDeniedError(
                message="User submitting request does not have permission to remove user from role."
            )

        await session.execute(
            delete(UserRole).where(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id,
            )
        )
        await session.commit()
