import { Avatar, AvatarImage } from "@/components/ui/avatar.jsx";
import { useSelector, useDispatch } from "react-redux";
import { apiClient } from "@/lib/api-client.js";
import { useEffect, useState } from "react";
import { GET_USER_IN_CHANNEL_ROUTE } from "@/utils/constants.js";
import { HiUserRemove } from "react-icons/hi";
import { RiLogoutBoxRLine } from "react-icons/ri";

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useSocket } from "@/context/SocketContext.jsx";

function MemberList({ channel }) {
  const socket = useSocket().current;
  const dispatch = useDispatch();
  const [memberList, setMemberList] = useState([]);
  const { currentUser } = useSelector((state) => state.user);

  useEffect(() => {
    const getMembers = async () => {
      try {
        const response = await apiClient.get(
          `${GET_USER_IN_CHANNEL_ROUTE}/${channel._id}`,
          { withCredentials: true }
        );
        setMemberList(response.data.members);
      } catch (error) {
        console.error("Error fetching members:", error);
      }
    };
    getMembers();
  }, [channel]);

  const handleRemoveMember = async (member) => {
    try {
      socket.emit("remove-member", {
        sender: currentUser,
        channelId: channel._id,
        memberId: [member._id],
        memberName: member.name,
      });
    } catch (error) {
      console.error("Error removing member:", error);
    }
  };

  const handleLeaveGroup = async (member) => {
    try {
      socket.emit("remove-member", {
        sender: currentUser,
        channelId: channel._id,
        memberId: [member._id],
        memberName: member.name,
      });
    } catch (error) {
      console.error("Error removing member:", error);
    }
  };

  return (
    <div className="flex flex-col gap-2">
      {memberList?.length > 0 &&
        memberList.map((member) => {
          return (
            <div
              key={member._id}
              className="py-2 px-2 flex rounded-md gap-5 items-center relative mr-5"
            >
              <div className="rounded-full flex items-center justify-center">
                <Avatar className="h-11 w-11">
                  <AvatarImage src={member.profilePicture} />
                </Avatar>
              </div>
              <span className="text-lg font-medium">{member.name}</span>
              {member._id === currentUser._id &&
                !channel?.admin?.includes(currentUser?._id) && (
                  <AlertDialog>
                    <AlertDialogTrigger className="absolute right-0 hover:bg-[#00000022] bg-[#00000022] rounded-full p-2 cursor-pointer">
                      <RiLogoutBoxRLine className="text-black text-xl" />
                    </AlertDialogTrigger>
                    <AlertDialogContent className="w-[50vw] bg-white">
                      <AlertDialogHeader>
                        <AlertDialogTitle className="text-2xl font-medium text-gray-800 text-center">
                          Are you absolutely sure?
                        </AlertDialogTitle>
                        <AlertDialogDescription className="text-lg text-gray-800">
                          {`This action cannot be undone. This will permanently remove YOU from this group.`}
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel className="hover:bg-[#E4E4E4] transition-all duration-100 text-lg">
                          Cancel
                        </AlertDialogCancel>
                        <AlertDialogAction
                          className="text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-100 text-lg"
                          onClick={() => handleLeaveGroup(member)}
                        >
                          Continue
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                )}
              {channel?.admin?.includes(currentUser?._id) &&
                member._id !== currentUser._id && (
                  <AlertDialog>
                    <AlertDialogTrigger className="absolute right-0 hover:bg-[#00000022] bg-[#00000022] rounded-full p-2 cursor-pointer">
                      <HiUserRemove className="text-black text-xl" />
                    </AlertDialogTrigger>
                    <AlertDialogContent className="w-[50vw] bg-white">
                      <AlertDialogHeader>
                        <AlertDialogTitle className="text-2xl font-medium text-gray-800 text-center">
                          Are you absolutely sure?
                        </AlertDialogTitle>
                        <AlertDialogDescription className="text-lg text-gray-800">
                          {`This action cannot be undone. This will permanently remove ${member.name} from this group.`}
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel className="hover:bg-[#E4E4E4] transition-all duration-100 text-lg">
                          Cancel
                        </AlertDialogCancel>
                        <AlertDialogAction
                          className="text-white bg-[#26597C] hover:bg-[#1d4762] transition-all duration-100 text-lg"
                          onClick={() => handleRemoveMember(member)}
                        >
                          Continue
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                )}
            </div>
          );
        })}
    </div>
  );
}

export default MemberList;
