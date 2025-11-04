import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { useState } from "react";
import { FaPlus } from "react-icons/fa";
import { AiOutlineUserAdd } from "react-icons/ai";
import { Input } from "@/components/ui/input";
import Lottie from "react-lottie";
import { animationDefaultOption } from "@/lib/utils";
import { apiClient } from "@/lib/api-client";
import { SEARCH_CONTACTS_ROUTE } from "@/utils/constants";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarImage } from "@/components/ui/avatar";
import { useDispatch, useSelector } from "react-redux";
import {
  setSelectedChatType,
  setSelectedChatData,
} from "@/redux/chat/chatSlice";
import { Button } from "@/components/ui/button";

function NewDM({ displayText, className }) {
  const [openNewContactModal, setOpenNewContactModal] = useState(false);
  const [searchedContacts, setSearchedContacts] = useState([]);
  const dispatch = useDispatch();

  const { selectedChatData, selectedChatType } = useSelector(
    (state) => state.chat
  );

  const searchContacts = async (searchTerm) => {
    try {
      if (searchTerm.length > 0) {
        const response = await apiClient.post(
          SEARCH_CONTACTS_ROUTE,
          { searchTerm },
          { withCredentials: true }
        );
        if (response.status === 200 && response.data.contacts) {
          setSearchedContacts(response.data.contacts);
        }
      } else {
        setSearchedContacts([]);
      }
    } catch (error) {
      console.log(error);
    }
  };

  const selectNewContact = (contact) => {
    dispatch(setSelectedChatType("contact"));
    dispatch(setSelectedChatData(contact));
    setOpenNewContactModal(false);
    setSearchedContacts([]);
  };

  return (
    <>
      <Button
        className={
          className ||
          "group cursor-pointer bg-[#F8F8D5] border-2 border-black ml-2 transition-all duration-300 hover:bg-[#cfcfa1]"
        }
        onClick={() => setOpenNewContactModal(true)}
      >
        {displayText ? (
          <span className="text-center text-lg text-white">{displayText}</span>
        ) : (
          <AiOutlineUserAdd
            size={30}
            className="cursor-pointer text-start font-light text-black transition-all duration-300"
          />
        )}
      </Button>
      <Dialog open={openNewContactModal} onOpenChange={setOpenNewContactModal}>
        <DialogContent className="flex h-[600px] w-[500px] flex-col border-none bg-[#ffffff]">
          <DialogHeader>
            <DialogTitle className="text-center text-xl">
              Select a contact
            </DialogTitle>
            <DialogDescription></DialogDescription>
          </DialogHeader>
          <div>
            <Input
              placeholder="Search contacts"
              className="rounded-lg border-2 border-black-20 bg-[#ebebeb] p-6 transition-all duration-100 focus:bg-white"
              onChange={(e) => searchContacts(e.target.value)}
            />
          </div>
          {searchedContacts?.length > 0 && (
            <ScrollArea className="h-full">
              <div className="flex flex-col gap-5">
                {searchedContacts.map((contact) => (
                  <div
                    key={contact._id}
                    className="flex cursor-pointer items-center gap-3"
                    onClick={() => selectNewContact(contact)}
                  >
                    <div className="relative h-12 w-12">
                      <Avatar className="h-12 w-12 overflow-hidden rounded-full">
                        <AvatarImage
                          src={contact.profilePicture}
                          alt="profile"
                          className="h-full w-full bg-black object-cover"
                        />
                      </Avatar>
                    </div>
                    <div className="flex flex-col">
                      <span>{contact.name ? contact.name : contact.email}</span>
                      <span className="text-xs">{contact.email}</span>
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}
          {searchedContacts.length <= 0 && (
            <div className="flex-1 flex-col items-center justify-center transition-all duration-1000 md:flex">
              <Lottie
                isClickToPauseDisabled={true}
                height={100}
                width={100}
                options={animationDefaultOption}
              />
              <div className="flex flex-col items-center gap-5 text-center text-xl text-black text-opacity-80 transition-all duration-300 lg:text-2xl">
                <h3 className="">Search new Contact</h3>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}

export default NewDM;
